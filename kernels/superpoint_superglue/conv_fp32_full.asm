// ============================================================================
// conv_fp32_full.asm  --  3x3 FP32 conv, ONE FULL OUTPUT CHANNEL per launch
// ----------------------------------------------------------------------------
// Computes a complete output channel for a resident input band in ONE launch:
// loops rows x column-tiles x channel-groups internally (no host width-tiling
// or column stitching). Width-tiling is in the kernel (column-tile loop);
// channel-group tiling (G<=14) handles any Cin; bias is the all-ones channel.
//
// Input rows are FULL-WIDTH: each padded input row spans RW bytes
// (= num_col_tiles*512), 1px left/right border = 0. Tap (kr,kc) for output col
// (ct*128+lane) reads padded byte offset (ct*128+kc)*4 + lane*4 of row+kr, i.e.
// base + kr*RW + kc*4 ; col-tile advances the base by 512. R_ACC accumulates
// over all channel-groups for one (row, col-tile), then relu + store.
//
// NOTE on memory: a full real-res activation exceeds the 2MB XMEM, so the host
// (cache unit on HW) still streams ROW-BANDS into XMEM; this kernel computes the
// full width and all band rows for one channel per launch -- only band
// streaming + one launch/channel remain on the host, no width work/stitching.
//
// TIGHT INPUT PACKING: input rows use a tight stride RW_in=(W+2)*4 (no rounding
// to whole 128-lane tiles) -> the resident input has no column padding, ~16-40%
// smaller, so a streamed band holds more rows. Output stays 512-tile-aligned
// (STR writes 512B), row stride RW_out=num_col_tiles*512. The last column-tile's
// 128-lane read spills past the packed input row into the next row but only
// affects output cols >= W (in the output's padding tile, never read); add ONE
// guard input row at the band end so the last row's spill stays in bounds.
// Caveat: valid for per-layer eval (host lays out tight input); a chained
// pipeline keeps 512-padded I/O (store granularity) unless the host re-packs.
//
// Register / memory contract:
//   CR0=0, CR1=1
//   CR2=IN, CR3=OUT, CR4=in_chan_stride (Hp*RW_in), CR5=RW_in ((W+2)*4, tight),
//   CR6=H (band rows), CR7=512 (col-tile stride), CR8=RW_out (num_col_tiles*512),
//   CR9=weights_base, CR10=128, CR11=G, CR12=ngroups, CR13=group_weight_stride,
//   CR14=num_col_tiles, CR15=dtype.  (tap row-wrap RW_in-8 computed into lr14.)
//   Input planes Ctot at IN+ci*in_chan_stride, tight RW_in-wide rows (+1 guard row).
//   Output plane: H rows x RW_out bytes, valid cols 0..W-1.
// ============================================================================

    SET                 lr15 cr10 ;;        // 128 (valid lanes for relu)
    SET                 lr0 cr0 ;;           // 0
    SET                 lr1 cr0 ;;           // INPUT row byte offset = 0 (stride RW_in=CR5)
    SET                 lr2 cr0 ;;           // row counter = 0
    SET                 lr13 cr0 ;;          // OUTPUT row byte offset = 0 (stride RW_out=CR8)
    SET                 lr14 cr5 ;;
    SUB                 lr14 lr14 8 ;;       // lr14 = RW_in - 8 (tap row-wrap delta)

row_loop:
    SET                 lr8 cr0 ;;           // col-tile counter = 0
    SET                 lr9 cr0 ;;           // col-tile byte offset = 0

coltile_loop:
    RESET_ACC ;;
    SET                 lr3 cr2 ;;           // channel-0 tap base = IN
    ADD                 lr3 lr3 lr1 ;;       // + row offset
    ADD                 lr3 lr3 lr9 ;;       // + col-tile offset
    SET                 lr7 cr9 ;;           // weight base
    SET                 lr10 cr0 ;;          // group counter = 0

group_loop:
    LDR_MULT_REG        r0 lr7 cr0 ;;        // R0 = weights[group]
    SET                 lr4 cr0 ;;
    SUB                 lr4 lr4 1 ;;         // weight lane idx = -1 (reset PER GROUP)
    SET                 lr5 cr0 ;;           // cin-in-group counter = 0

cin_loop:
    ADD                 lr6 lr3 cr0 ;;       // tap addr = channel base (tap (0,0))
    ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD lr6 lr6 4 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD lr6 lr6 4 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD lr6 lr6 lr14 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD lr6 lr6 4 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD lr6 lr6 4 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD lr6 lr6 lr14 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD lr6 lr6 4 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD lr6 lr6 4 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD                 lr3 lr3 cr4 ;;       // next input channel plane
    ADD                 lr5 lr5 cr1 ;;
    BLT                 lr5 cr11 cin_loop ;;
    ADD                 lr7 lr7 cr13 ;;      // weight base += group stride
    ADD                 lr10 lr10 cr1 ;;
    BLT                 lr10 cr12 group_loop ;;

    ACTIVATE            lr15 relu ;;
    ADD                 lr11 lr13 lr9 ;;     // out offset = OUTPUT row (RW_out) + col-tile
    STR_POST_AAQ_REG    lr11 cr3 ;;
    ADD                 lr9 lr9 cr7 ;;       // next col-tile (+512)
    ADD                 lr8 lr8 cr1 ;;
    BLT                 lr8 cr14 coltile_loop ;;

    ADD                 lr1 lr1 cr5 ;;       // next INPUT row (+RW_in)
    ADD                 lr13 lr13 cr8 ;;     // next OUTPUT row (+RW_out)
    ADD                 lr2 lr2 cr1 ;;
    BLT                 lr2 cr6 row_loop ;;

    BKPT ;;
