// ============================================================================
// convPa.asm  --  SuperPoint convPa: 128->256, 3x3, ReLU
// ----------------------------------------------------------------------------
// One full output channel per launch (loops rows x col-tiles x channel-groups).
// Base kernel: conv_fp32_full. Run 256 times (one per
// output channel), advancing OUT (CR3) and weights_base (CR9) by the host.
// Channel config: Cin'=129 (incl. bias ones-channel), G=13, ngroups=10,
//   group_weight_stride=468 (CR13), G(CR11)=13, ngroups(CR12)=10.
// Set CR2=IN, CR3=OUT, CR4=in_chan_stride, CR5=RW_in, CR6=H, CR7=512,
//   CR8=RW_out, CR9=weights_base, CR10=128, CR14=num_col_tiles, CR15=dtype.
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
