// ============================================================================
// conv_fp32.asm  --  3x3 FP32 convolution from scratch, VLIW-packed
// ----------------------------------------------------------------------------
// One output channel = relu( sum_cin sum_(kr,kc) W[cin][kr][kc] * in[cin][y+kr][x+kc] ).
// FP32 wide-vector mode (masks off -> zero-padded input). Each tap is one
// PACKED compound cycle: { LR addr-update ; LR weight-index++ ; LDR_CYCLIC
// shifted input ; MULT.VE.CYCLIC(weight) ; ACC } -> MULT and ACC busy every
// tap cycle (high ALU utilization). Spatial neighbours are constant-offset
// shifted contiguous loads; arbitrary FP32 weights come from R0 via fixed_idx.
//
// Layout (precomputed offline): input is Cin zero-padded planes, each Hp rows x
// 128 cols (1-px border = 0). Plane cin at IN + cin*chan_stride. Output row y's
// 3x3 window uses padded rows y,y+1,y+2 (center y+1 = original row y). Weights
// for this output channel = Cin*9 floats in R0, order [cin][tap], tap row-major.
//
// Register / memory contract:
//   CR0=0, CR1=1
//   CR2  = IN base (padded input, channel 0)
//   CR3  = OUT base (this output channel, num_rows x 128)
//   CR4  = chan_stride (bytes per input plane = Hp*512)
//   CR5  = rowbytes (= 512; also output row stride)
//   CR6  = num_rows (output rows)
//   CR7  = Cin (input channels; Cin*9 <= 128)
//   CR8  = 504 (row-wrap addr delta = rowbytes - 2*elem = 512-8)
//   CR9  = weights_addr (Cin*9 f32, order [cin][tap])
//   CR10 = 128 (valid lanes for relu)
//   CR15 = dtype sentinel (INT8)
//
// Output: OUT[y] = relu(conv) per row, 126 valid cols (lanes 0..125).
// ============================================================================

    SET                 lr10 cr10 ;;        // 128
    SET                 lr0 cr0 ;;           // 0
    LDR_MULT_REG        r0 lr0 cr9 ;;        // R0 = weights (prologue; stable in snapshot)
    SET                 lr1 cr0 ;;           // lr1 = y row byte offset = 0
    SET                 lr2 cr0 ;;           // lr2 = output row counter = 0

row_loop:
    RESET_ACC ;;
    SET                 lr3 cr2 ;;           // lr3 = channel base = IN
    ADD                 lr3 lr3 lr1 ;;       // + y*rowbytes
    SET                 lr4 cr0 ;;           // lr4 = weight fixed_idx (byte) -- start 0
    SUB                 lr4 lr4 1 ;;         // lr4 = -1 (each tap pre-increments +1; fixed_idx is a LANE index)
    SET                 lr5 cr0 ;;           // lr5 = cin counter = 0

cin_loop:
    ADD                 lr6 lr3 cr0 ;;       // lr6 = tap address = channel base (tap0)
    // tap (0,0)  delta 0
    ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    // tap (0,1)  +4
    ADD lr6 lr6 4 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    // tap (0,2)  +4
    ADD lr6 lr6 4 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    // tap (1,0)  +504 (next row, back 2 cols)
    ADD lr6 lr6 cr8 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    // tap (1,1)  +4
    ADD lr6 lr6 4 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    // tap (1,2)  +4
    ADD lr6 lr6 4 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    // tap (2,0)  +504
    ADD lr6 lr6 cr8 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    // tap (2,1)  +4
    ADD lr6 lr6 4 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    // tap (2,2)  +4
    ADD lr6 lr6 4 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;

    ADD                 lr3 lr3 cr4 ;;       // next input channel plane
    ADD                 lr5 lr5 cr1 ;;       // cin++
    BLT                 lr5 cr7 cin_loop ;;

    ACTIVATE            lr10 relu ;;         // POST_AAQ = relu(acc)
    STR_POST_AAQ_REG    lr1 cr3 ;;           // OUT[y] = result
    ADD                 lr1 lr1 cr5 ;;       // next output row
    ADD                 lr2 lr2 cr1 ;;
    BLT                 lr2 cr6 row_loop ;;

    BKPT ;;
