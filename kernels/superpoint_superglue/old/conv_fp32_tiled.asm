// ============================================================================
// conv_fp32_tiled.asm  --  3x3 FP32 conv, CHANNEL-GROUP TILED (any Cin)
// ----------------------------------------------------------------------------
// Same packed MAC as conv_fp32 (1 cycle/tap), but Cin is split into groups of
// G<=14 channels so the per-group weights (G*9 floats) fit R0. R_ACC is reset
// once per output row and ACCUMULATES across groups (no reset between groups),
// so this handles real SuperPoint layers (Cin up to 256). Bias is folded as an
// extra all-ones input channel (center-tap weight = bias) in the last group.
//
// Loops (nested is fine for a per-op kernel): row -> group -> cin-in-group;
// the 9 taps are unrolled. ~41 words, <=128 bank.
//
// Layout: Cin' (=Cin+1 with the bias ones-channel) zero-padded planes, plane ci
// at IN+ci*chan_stride. Weights row-major [group][cin-in-group][tap] in XMEM at
// weights_base; group g at weights_base + g*group_weight_stride.
//
// Register / memory contract:
//   CR0=0, CR1=1
//   CR2=IN, CR3=OUT (this out channel), CR4=chan_stride (Hp*512), CR5=512,
//   CR6=Hc (rows), CR8=504, CR9=weights_base, CR10=128,
//   CR11=G (channels per group, G*9<=128), CR12=num_groups (=ceil(Cin'/G)),
//   CR13=group_weight_stride (=G*9*4 bytes), CR15=dtype.
// Note: total input channels processed = G*num_groups (pad weights with 0 if
//   Cin' not a multiple of G).
// ============================================================================

    SET                 lr10 cr10 ;;        // 128
    SET                 lr0 cr0 ;;           // 0
    SET                 lr1 cr0 ;;           // rowoff = 0
    SET                 lr2 cr0 ;;           // row counter = 0

row_loop:
    RESET_ACC ;;
    SET                 lr3 cr2 ;;           // channel-0 base = IN
    ADD                 lr3 lr3 lr1 ;;       // + rowoff
    SET                 lr7 cr9 ;;           // weight base = weights_base
    SET                 lr9 cr0 ;;           // group counter = 0

group_loop:
    LDR_MULT_REG        r0 lr7 cr0 ;;        // R0 = weights[group] (addr=lr7)
    SET                 lr4 cr0 ;;
    SUB                 lr4 lr4 1 ;;         // weight lane idx = -1
    SET                 lr5 cr0 ;;           // cin-in-group counter = 0

cin_loop:
    ADD                 lr6 lr3 cr0 ;;       // tap addr = channel base (tap0)
    ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD lr6 lr6 4 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD lr6 lr6 4 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD lr6 lr6 cr8 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD lr6 lr6 4 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD lr6 lr6 4 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD lr6 lr6 cr8 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD lr6 lr6 4 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD lr6 lr6 4 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD                 lr3 lr3 cr4 ;;       // next input channel plane
    ADD                 lr5 lr5 cr1 ;;       // cin-in-group++
    BLT                 lr5 cr11 cin_loop ;;

    ADD                 lr7 lr7 cr13 ;;      // weight base += group stride
    ADD                 lr9 lr9 cr1 ;;       // group++
    BLT                 lr9 cr12 group_loop ;;

    ACTIVATE            lr10 relu ;;
    STR_POST_AAQ_REG    lr1 cr3 ;;           // OUT[row]
    ADD                 lr1 lr1 cr5 ;;       // next row
    ADD                 lr2 lr2 cr1 ;;
    BLT                 lr2 cr6 row_loop ;;

    BKPT ;;
