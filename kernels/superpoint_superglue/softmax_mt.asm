// ============================================================================
// softmax_mt.asm  --  multi-tile softmax over N = ntiles * 128 lanes
// ----------------------------------------------------------------------------
// Companion to softmax.asm (single 128-lane tile). Computes a numerically
// stable base-2 softmax across N (= 128*ntiles) lanes in FOUR passes over the
// input, exactly as on the whiteboard:
//
//   1. running element-wise max ACROSS tiles via ACC.MAX  ->  AGG max value_cr(-1)
//      ->  aaq0 = -global_max                                (one cycle per tile)
//   2. per tile: R_ACC = x - global_max ; ACTIVATE exp2 ; STR_POST_AAQ -> scratch
//   3. running element-wise sum of exps via ACC  ->  AGG sum inv
//      ->  aaq0 = 1 / global_sum                             (one cycle per tile)
//   4. per tile: R_ACC = exps * (1/global_sum) ; STR_ACC -> out
//
// Same base-2 convention as softmax.asm: caller pre-scales logits by log2(e)
// for natural-base softmax (so 2^(x*log2 e) = e^x -- exact).
//
// Register / memory contract (caller sets these):
//   CR0   = 0                  (zero base / index)
//   CR1   = 1                  (loop increment)
//   CR2   = in_addr            N x f32 logits, contiguous 128-lane tiles
//   CR3   = scratch_addr       ntiles * 512 B scratch (exp values)
//   CR4   = ones_addr          128 x f32 all-1.0
//   CR5   = out_addr           N x f32 output probabilities (ntiles * 512 B)
//   CR6   = ntiles             number of 128-lane tiles (N = ntiles * 128)
//   CR7   = float32_bits(-1.0) = 0xBF800000   (negate-the-max constant)
//   CR8   = 512                tile byte stride
//   CR9   = 128                valid lanes per tile (AGG reduction width)
//   CR10  = neginf_addr        128 x f32 all -3.4e38 (seed aaq0 for ACC.MAX)
//   CR15  = dtype sentinel (INT8)
//
// Assumes N is a clean multiple of 128. Output: out_addr[0..N) = softmax(logits).
// ============================================================================

    SET                 lr0 cr0 ;;          // 0
    SET                 lr5 cr9 ;;          // valid_lanes = 128

    // ---- aaq0 = -inf  (neutral seed for the 3-way ACC.MAX) ---------------
    LDR_MULT_REG        r0 lr0 cr10 ;;
    LDR_CYCLIC_MULT_REG lr0 cr4 lr0 ; MULT.EE r0 lr0 0 lr0 ; ACC.FIRST ;;
    AGG.FIRST           max value lr5 cr0 aaq0 ;;       // aaq0 = -inf

// ============ Pass 1: running element-wise max across tiles ===============
    SET                 lr1 cr0 ;;          // tile counter
    SET                 lr2 cr0 ;;          // input byte offset
    // tile 0: R_ACC = tile 0
    LDR_MULT_REG        r0 lr2 cr2 ;;
    LDR_CYCLIC_MULT_REG lr0 cr4 lr0 ; MULT.EE r0 lr0 0 lr0 ; ACC.FIRST ;;
    ADD lr2 lr2 cr8 ; ADD lr1 lr1 cr1 ;;
p1_loop:
    LDR_MULT_REG        r0 lr2 cr2 ;;
    LDR_CYCLIC_MULT_REG lr0 cr4 lr0 ; MULT.EE r0 lr0 0 lr0 ; ACC.MAX aaq0 ;;
    ADD lr2 lr2 cr8 ; ADD lr1 lr1 cr1 ;;
    BLT                 lr1 cr6 p1_loop ;;
    // reduce R_ACC -> scalar  -global_max
    AGG.FIRST           max value_cr lr5 cr7 aaq0 ;;    // aaq0 = -m

// ============ Pass 2: per tile, exp2(x - m) -> scratch ===================
    SET                 lr1 cr0 ;;
    SET                 lr2 cr0 ;;
    SET                 lr3 cr0 ;;          // scratch byte offset
p2_loop:
    LDR_MULT_REG        r0 lr2 cr2 ;;
    LDR_CYCLIC_MULT_REG lr0 cr4 lr0 ; MULT.EE r0 lr0 0 lr0 ; ACC.ADD_AAQ.FIRST aaq0 ;;
    ACTIVATE            lr5 exp2 ;;
    STR_POST_AAQ_REG    lr3 cr3 ;;
    ADD lr2 lr2 cr8 ; ADD lr3 lr3 cr8 ; ADD lr1 lr1 cr1 ;;
    BLT                 lr1 cr6 p2_loop ;;

// ============ Pass 3: running sum of exps -> aaq0 = 1/sum =================
    SET                 lr1 cr0 ;;
    SET                 lr3 cr0 ;;
    // tile 0: R_ACC = exps tile 0
    LDR_MULT_REG        r0 lr3 cr3 ;;
    LDR_CYCLIC_MULT_REG lr0 cr4 lr0 ; MULT.EE r0 lr0 0 lr0 ; ACC.FIRST ;;
    ADD lr3 lr3 cr8 ; ADD lr1 lr1 cr1 ;;
p3_loop:
    LDR_MULT_REG        r0 lr3 cr3 ;;
    LDR_CYCLIC_MULT_REG lr0 cr4 lr0 ; MULT.EE r0 lr0 0 lr0 ; ACC ;;
    ADD lr3 lr3 cr8 ; ADD lr1 lr1 cr1 ;;
    BLT                 lr1 cr6 p3_loop ;;
    AGG.FIRST           sum inv lr5 cr0 aaq0 ;;         // aaq0 = 1/sum

// ============ Pass 4: per tile, softmax = exp * (1/sum) ==================
    SET                 lr1 cr0 ;;
    SET                 lr3 cr0 ;;
    SET                 lr4 cr0 ;;          // out byte offset
p4_loop:
    LDR_CYCLIC_MULT_REG lr3 cr3 lr0 ; MULT.VE.AAQ lr0 0 lr0 aaq0 ; ACC.FIRST ;;
    STR_ACC_REG         lr4 cr5 ;;
    ADD lr3 lr3 cr8 ; ADD lr4 lr4 cr8 ; ADD lr1 lr1 cr1 ;;
    BLT                 lr1 cr6 p4_loop ;;

    BKPT ;;
