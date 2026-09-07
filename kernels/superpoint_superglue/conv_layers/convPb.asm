// ============================================================================
// convPb.asm  --  SuperPoint convPb: 256->65, 1x1, no ReLU
// ----------------------------------------------------------------------------
// One full output channel per launch (loops rows x col-tiles x channel-groups).
// Base kernel: conv1x1. Run 65 times (one per
// output channel), advancing OUT (CR3) and weights_base (CR9) by the host.
// Channel config: Cin'=257 (incl. bias ones-channel), G=13, ngroups=20,
//   group_weight_stride=52 (CR13), G(CR11)=13, ngroups(CR12)=20.
// Set CR2=IN, CR3=OUT, CR4=in_chan_stride, CR5=RW_in, CR6=H, CR7=512,
//   CR8=RW_out, CR9=weights_base, CR10=128, CR14=num_col_tiles, CR15=dtype.
// ============================================================================

    SET                 lr15 cr10 ;;        // 128
    SET                 lr0 cr0 ;;           // 0
    SET                 lr1 cr0 ;;           // input row offset = 0
    SET                 lr2 cr0 ;;           // row counter = 0
    SET                 lr13 cr0 ;;          // output row offset = 0

row_loop:
    SET                 lr8 cr0 ;;           // col-tile counter = 0
    SET                 lr9 cr0 ;;           // col-tile byte offset = 0

coltile_loop:
    RESET_ACC ;;
    SET                 lr3 cr2 ;;           // channel-0 base = IN
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
    ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr3 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD                 lr3 lr3 cr4 ;;       // next input channel plane
    ADD                 lr5 lr5 cr1 ;;
    BLT                 lr5 cr11 cin_loop ;;
    ADD                 lr7 lr7 cr13 ;;      // weight base += group stride
    ADD                 lr10 lr10 cr1 ;;
    BLT                 lr10 cr12 group_loop ;;

    ADD                 lr11 lr13 lr9 ;;     // out offset = out-row + col-tile
    STR_ACC_REG         lr11 cr3 ;;          // raw conv+bias (no ReLU)
    ADD                 lr9 lr9 cr7 ;;       // next col-tile (+512)
    ADD                 lr8 lr8 cr1 ;;
    BLT                 lr8 cr14 coltile_loop ;;

    ADD                 lr1 lr1 cr5 ;;       // next input row
    ADD                 lr13 lr13 cr8 ;;     // next output row
    ADD                 lr2 lr2 cr1 ;;
    BLT                 lr2 cr6 row_loop ;;

    BKPT ;;
