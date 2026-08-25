{#- ==========================================================================
    maxpool2d_halve.asm -- 2x2 stride-2 max-pool, FP32 wide-vector mode

    out[c, y, x] = max( in[c, 2y,   2x], in[c, 2y,   2x+1],
                        in[c, 2y+1, 2x], in[c, 2y+1, 2x+1] )

    SuperPoint applies this three times (after conv1b, conv2b, conv3b). One
    launch produces the whole (C, H//2, W//2) output.

    WHY THE FOUR TAPS COST NOTHING EXTRA:
      XMEM operands are ROW numbers -- a load addresses whole 128-element rows
      and cannot shift by one element. MULT.RC.* instead reads R_CYCLIC at an
      arbitrary ELEMENT index and may cross slot boundaries, so the horizontal
      shift happens inside the register: the two vertically-neighbouring rows
      occupy R_CYCLIC slots 0/1 and 2/3, and dx becomes a +1 step on the read
      index. `MULT.RC.VE rc, cr1` is the identity move (R_CYCLIC x 1.0), so a
      tap is one MULT plus one ACC.MAX.

      ACC.MAX.FIRST seeds R_ACC from the first tap, so no -inf seed vector is
      needed and nothing carries over from the previous tile.

    THE DECIMATION (this is the whole difficulty):
      Four taps give the STRIDE-1 2x2 maximum at every column; a stride-2 pool
      wants only the even ones. ACC.STRIDE does exactly that decimation -- but
      it writes MULT_RES into R_ACC, OVERWRITING, so it cannot itself take a
      max and the max must already be finished. Hence the round trip:

        1. four taps -> R_ACC = stride-1 2x2 max at every column
        2. ACTIVATE identity + STR_POST_AAQ  -> a scratch row
        3. reload the scratch, MULT.RC.VE x 1.0, ACC.STRIDE 64 on off

      `ACC.STRIDE 64 on off` reads MULT_RES as two rows of 64, takes every
      second element of each, and writes the 64 survivors CONTIGUOUSLY at
      R_ACC[(offset % 4) * 32]. That is exactly lanes 0,2,...,126 packed down
      to 64 -- the even columns, which are the stride-2 outputs.

      BOTH HALVES ARE STAGED BEFORE EITHER ACC.STRIDE. Half B's ACC.MAX.FIRST
      overwrites all 128 R_ACC elements, so it would destroy half A's result if
      half A had already been decimated into R_ACC[0:64]. Staging both, then
      decimating both, works because ACC.STRIDE leaves the R_ACC indices it does
      NOT write untouched: half A lands at base 0 (offset 0) and half B at base
      64 (offset 2), filling one 128-lane output row between them.

    TILING:
      One output XMEM row is 128 output columns, so it spans 256 input columns
      = two full-width input tiles ("halves"). Output tile ot reads input tiles
      2*ot (half A) and 2*ot+1 (half B).

      Each spatial row carries ONE GUARD TILE past the tiles the halves read,
      filled with -FLT_MAX. It exists because the dx=1 tap reads element 128 of
      its slot pair, i.e. the first element of the NEXT tile of the SAME spatial
      row -- so both slots of a pair are loaded, and the last half of the last
      output tile still needs a tile after it. -FLT_MAX can never win a maximum,
      so the guard is inert.

      Columns past W inside a partly-filled tile are likewise -FLT_MAX. They
      never reach a kept output lane (output lane j < W//2 reads input columns
      2j and 2j+1, both < W), but filling them with the maximum's identity keeps
      the discarded lanes finite and debuggable rather than garbage.

      NO VERTICAL BORDER: padding is 0, so rows 2y and 2y+1 are always real
      image rows and there is no top/bottom special case. An odd H or W simply
      drops the last row/column, which is what nn.MaxPool2d(2, 2) does.

    R_CYCLIC SLOTS (per half):
      slot 0 (idx   0) <- spatial row 2y,   tile it
      slot 1 (idx 128) <- spatial row 2y,   tile it+1     (supplies element 128)
      slot 2 (idx 256) <- spatial row 2y+1, tile it
      slot 3 (idx 384) <- spatial row 2y+1, tile it+1     (supplies element 384)

      Taps read rc = 0, 1, 256, 257. LDR_CYCLIC_MULT_REG's index is restricted
      to slot boundaries {0,128,256,384}; MULT.RC.VE's rc_idx is not, which is
      what rc=1 and rc=257 rely on. R_CYCLIC contents are read from the
      start-of-word SNAPSHOT, so every load lands at least one word before the
      multiply that consumes it.

    CR map (set by the harness; CR0/CR1 are READ-ONLY hardware constants):
      CR0  = 0                        CR1  = 1  (-> 1.0 scalar; every +1)
      CR2  = INPUT_BASE   (rows)      CR3  = OUTPUT_BASE  (rows)
      CR4  = SCRATCH_BASE (rows; 2 rows)
      CR5  = IN_ROW_STRIDE   -- input XMEM rows per spatial row, guard included
      CR6  = 2 * IN_ROW_STRIDE -- advance from spatial row 2y to 2(y+1)
      CR7  = OUT_TILES_PER_ROW        CR8  = OUT_HEIGHT = H // 2
      CR9  = CHANNELS                 CR10 = 128
      CR11 = IN_PLANE_STRIDE = H * IN_ROW_STRIDE
      CR15 = dstructure (valid_elements = 128)

    LR uses 3 sub-slots; ";;" ends a VLIW word, ";" separates sub-instructions.
    Slot order within a word is LR -> LOAD -> MULT -> ACC -> AAQ -> STORE ->
    COND. That is why a load can share a word with the LR add that positions it
    (LDR offsets read LIVE), and why the store's row counter is incremented in
    the word AFTER the store.
========================================================================== -#}

{%- set lr_zero  = "lr0"  -%}  {#- 0: mask_shift, slot-0 index, tap (0,0), scratch row A, ACC.STRIDE base 0 -#}
{%- set lr_c     = "lr1"  -%}  {#- channel counter -#}
{%- set lr_out   = "lr2"  -%}  {#- running OUTPUT row counter (one row per store) -#}
{%- set lr_cbase = "lr3"  -%}  {#- first input row of this channel's plane -#}
{%- set lr_oy    = "lr4"  -%}  {#- output spatial row -#}
{%- set lr_rowb  = "lr5"  -%}  {#- input row of spatial row 2*oy, tile 0 -#}
{%- set lr_ot    = "lr6"  -%}  {#- output tile counter -#}
{%- set lr_addr  = "lr7"  -%}  {#- walking input row address -#}
{%- set lr_s1    = "lr8"  -%}  {#- 128: R_CYCLIC slot-1 index -#}
{%- set lr_s2    = "lr9"  -%}  {#- 256: R_CYCLIC slot-2 index AND tap (1,0) -#}
{%- set lr_s3    = "lr10" -%}  {#- 384: R_CYCLIC slot-3 index -#}
{%- set lr_one   = "lr11" -%}  {#- 1: tap (0,1) AND scratch row B offset -#}
{%- set lr_s2p1  = "lr12" -%}  {#- 257: tap (1,1) -#}
{%- set lr_half  = "lr13" -%}  {#- 2: ACC.STRIDE offset selecting R_ACC base 64 -#}
{%- set lr_itile = "lr14" -%}  {#- address of (spatial row 2*oy, input tile 2*ot) -#}

    SET {{lr_zero}} cr0 ;
    SET {{lr_c}}    cr0 ;
    SET {{lr_out}}  cr0 ;;
    SET {{lr_cbase}} cr0 ;
    SET {{lr_s1}}    cr10 ;
    SET {{lr_one}}   cr1 ;;                              {#- 0, 128, 1 -#}
    ADD {{lr_s2}} {{lr_s1}} {{lr_s1}} ;;                 {#- 256 (snapshot: lr_s1 set last word) -#}
    ADD {{lr_s3}} {{lr_s2}} {{lr_s1}} ;
    ADD {{lr_s2p1}} {{lr_s2}} cr1 ;
    ADD {{lr_half}} {{lr_one}} cr1 ;;                    {#- 384, 257, 2 -#}

chan_loop:
    SET {{lr_oy}} cr0 ;
    ADD {{lr_rowb}} {{lr_cbase}} cr0 ;;                  {#- rowb = this channel's plane base -#}

row_loop:
    SET {{lr_ot}} cr0 ;
    ADD {{lr_itile}} {{lr_rowb}} cr0 ;;                  {#- input tile 0 of spatial row 2*oy -#}

tile_loop:
{#- ---- HALF A: input tiles 2*ot and 2*ot+1 -------------------------------- -#}
    ADD {{lr_addr}} {{lr_itile}} cr0 ;
    LDR_CYCLIC_MULT_REG {{lr_addr}} cr2 {{lr_zero}} ;;   {#- slot0 <- row 2y,   tile it -#}
    INC {{lr_addr}} 1 ;
    LDR_CYCLIC_MULT_REG {{lr_addr}} cr2 {{lr_s1}} ;;     {#- slot1 <- row 2y,   tile it+1 -#}
    ADD {{lr_addr}} {{lr_itile}} cr5 ;
    LDR_CYCLIC_MULT_REG {{lr_addr}} cr2 {{lr_s2}} ;;     {#- slot2 <- row 2y+1, tile it -#}
    INC {{lr_addr}} 1 ;
    LDR_CYCLIC_MULT_REG {{lr_addr}} cr2 {{lr_s3}} ;;     {#- slot3 <- row 2y+1, tile it+1 -#}
    MULT.RC.VE {{lr_zero}}  cr1 0 {{lr_zero}} cr15 ; ACC.MAX.FIRST ;;   {#- dy 0 dx 0 -#}
    MULT.RC.VE {{lr_one}}   cr1 0 {{lr_zero}} cr15 ; ACC.MAX ;;         {#- dy 0 dx 1 -#}
    MULT.RC.VE {{lr_s2}}    cr1 0 {{lr_zero}} cr15 ; ACC.MAX ;;         {#- dy 1 dx 0 -#}
    MULT.RC.VE {{lr_s2p1}}  cr1 0 {{lr_zero}} cr15 ; ACC.MAX ;
    ACTIVATE.QUANTIZE identity cr15 ;
    STR_POST_AAQ_REG {{lr_zero}} cr4 ;;                  {#- dy 1 dx 1 | stage half A -#}

{#- ---- HALF B: input tiles 2*ot+1 and 2*ot+2 ------------------------------ -#}
    ADD {{lr_addr}} {{lr_itile}} cr1 ;
    LDR_CYCLIC_MULT_REG {{lr_addr}} cr2 {{lr_zero}} ;;   {#- slot0 <- row 2y,   tile it+1 -#}
    INC {{lr_addr}} 1 ;
    LDR_CYCLIC_MULT_REG {{lr_addr}} cr2 {{lr_s1}} ;;     {#- slot1 <- row 2y,   tile it+2 -#}
    ADD {{lr_addr}} {{lr_itile}} cr5 ;;                  {#- row 2y+1, tile it -#}
    INC {{lr_addr}} 1 ;
    LDR_CYCLIC_MULT_REG {{lr_addr}} cr2 {{lr_s2}} ;;     {#- slot2 <- row 2y+1, tile it+1 -#}
    INC {{lr_addr}} 1 ;
    LDR_CYCLIC_MULT_REG {{lr_addr}} cr2 {{lr_s3}} ;;     {#- slot3 <- row 2y+1, tile it+2 -#}
    MULT.RC.VE {{lr_zero}}  cr1 0 {{lr_zero}} cr15 ; ACC.MAX.FIRST ;;
    MULT.RC.VE {{lr_one}}   cr1 0 {{lr_zero}} cr15 ; ACC.MAX ;;
    MULT.RC.VE {{lr_s2}}    cr1 0 {{lr_zero}} cr15 ; ACC.MAX ;;
    MULT.RC.VE {{lr_s2p1}}  cr1 0 {{lr_zero}} cr15 ; ACC.MAX ;
    ACTIVATE.QUANTIZE identity cr15 ;
    STR_POST_AAQ_REG {{lr_one}} cr4 ;;                   {#- stage half B -#}

{#- ---- decimate both halves into one output row --------------------------- -#}
    LDR_CYCLIC_MULT_REG {{lr_zero}} cr4 {{lr_zero}} ;;   {#- slot0 <- scratch A -#}
    LDR_CYCLIC_MULT_REG {{lr_one}}  cr4 {{lr_s1}} ;;     {#- slot1 <- scratch B -#}
    MULT.RC.VE {{lr_zero}} cr1 0 {{lr_zero}} cr15 ;
    ACC.STRIDE 64 on off {{lr_zero}} ;;                  {#- even columns of A -> R_ACC[0:64] -#}
    MULT.RC.VE {{lr_s1}} cr1 0 {{lr_zero}} cr15 ;
    ACC.STRIDE 64 on off {{lr_half}} ;;                  {#- even columns of B -> R_ACC[64:128] -#}
    ACTIVATE.QUANTIZE identity cr15 ;
    STR_POST_AAQ_REG {{lr_out}} cr3 ;;                   {#- OUT[c, oy, ot] -#}
    ADD {{lr_out}} {{lr_out}} cr1 ;
    ADD {{lr_ot}} {{lr_ot}} cr1 ;
    INC {{lr_itile}} 2 ;;                                {#- next output tile spans two more input tiles -#}
    BLT {{lr_ot}} cr7 tile_loop ;;

    ADD {{lr_rowb}} {{lr_rowb}} cr6 ;                    {#- next output row: += 2 * IN_ROW_STRIDE -#}
    ADD {{lr_oy}} {{lr_oy}} cr1 ;;
    BLT {{lr_oy}} cr8 row_loop ;;

    ADD {{lr_cbase}} {{lr_cbase}} cr11 ;                 {#- next channel plane -#}
    ADD {{lr_c}} {{lr_c}} cr1 ;;
    BLT {{lr_c}} cr9 chan_loop ;;

end:
    BKPT ;;
