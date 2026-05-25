// ============================================================================
// superpoint.asm  --  SINGLE-FILE fused SuperPoint detector (conv+detect+thresh)
//                     one launch, FP32, LOOPS BUT NOT NESTED
// ----------------------------------------------------------------------------
// One .asm, one launch (emulator IMEM = 1024 words; this overflows the 128-word
// hardware bank -- emulator-only, the bank-swap pipeline is the HW-faithful form).
//
// Config baked by unrolling: Cin=2 input channels, Nc=4 detector channels, 3x3.
//
// Phase A (conv): a SINGLE flat loop over (output-channel, row). Cin*9 taps are
//   UNROLLED (straight-line packed MACs). Row advance is the loop; channel
//   advance is a WRAP CONDITIONAL (forward branch + weight reload) -- not a
//   nested loop. Writes Nc channel-major relu(conv) planes.
// Phase B (detect): a SINGLE flat loop over tiles; the Nc channels are UNROLLED.
//   conf = max_c logit_c ; keep = relu(conf - tau). Writes dense per-cell map.
//
// Layout: conv input = Cin zero-padded planes (Hp x 128), plane ci at IN+ci*CS.
//   conv/detect logits = Nc planes (Hc x 128), plane oc at OUTB+oc*DCS, row y +y*512.
//   detect keep-map reuses the (now-unused) weight region (CR9) at +tile*512.
//   const planes contiguous at PB: ones@PB, -inf@PB+512, -tau@PB+1024.
//
// Register / memory contract (all 16 CRs used):
//   CR0=0, CR1=1
//   CR2=IN, CR3=OUTB, CR4=CS (Hp*512), CR5=512, CR6=Hc, CR7=Nc,
//   CR8=504, CR9=WB (weights; also detect-output base), CR10=128,
//   CR11=cin_advance (CS-1032), CR12=weight_stride (Cin*9*4),
//   CR13=DCS (Hc*512), CR14=PB (const-planes base), CR15=dtype.
//   Weights in XMEM at WB, [oc][cin][tap] (Cin*9 f32 per channel).
// ============================================================================

    SET                 lr10 cr10 ;;        // 128
    SET                 lr0 cr0 ;;           // 0
    LDR_MULT_REG        r0 lr0 cr9 ;;        // R0 = weights[oc=0]
    SET                 lr1 cr0 ;;           // rowoff = 0
    SET                 lr2 cr0 ;;           // row counter = 0
    SET                 lr3 cr0 ;;           // oc counter = 0
    SET                 lr7 cr9 ;;           // weight base = WB
    SET                 lr8 cr0 ;;           // output channel base = oc*DCS = 0

A_loop:
    RESET_ACC ;;
    SET                 lr6 cr2 ;;           // tap addr = IN
    ADD                 lr6 lr6 lr1 ;;       // + rowoff  (channel0, tap0)
    SET                 lr4 cr0 ;;
    SUB                 lr4 lr4 1 ;;         // weight lane idx = -1
    // ---- cin0 (9 taps) ----
    ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD lr6 lr6 4 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD lr6 lr6 4 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD lr6 lr6 cr8 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD lr6 lr6 4 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD lr6 lr6 4 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD lr6 lr6 cr8 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD lr6 lr6 4 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD lr6 lr6 4 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    // ---- cin1 (9 taps; tap0 jumps to next channel base) ----
    ADD lr6 lr6 cr11 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD lr6 lr6 4 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD lr6 lr6 4 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD lr6 lr6 cr8 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD lr6 lr6 4 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD lr6 lr6 4 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD lr6 lr6 cr8 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD lr6 lr6 4 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;
    ADD lr6 lr6 4 ; ADD lr4 lr4 1 ; LDR_CYCLIC_MULT_REG lr6 cr0 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr4 ; ACC ;;

    ACTIVATE            lr10 relu ;;
    ADD                 lr9 lr8 lr1 ;;       // out offset = oc*DCS + rowoff
    STR_POST_AAQ_REG    lr9 cr3 ;;           // OUT[oc][row] = relu(conv)
    ADD                 lr1 lr1 cr5 ;;       // rowoff += 512
    ADD                 lr2 lr2 cr1 ;;       // row++
    BLT                 lr2 cr6 A_loop ;;    // same channel: next row
    // ---- channel wrap (forward branch + reload; not a nested loop) -------
    SET                 lr1 cr0 ;;           // rowoff = 0
    SET                 lr2 cr0 ;;           // row = 0
    ADD                 lr3 lr3 cr1 ;;       // oc++
    ADD                 lr7 lr7 cr12 ;;      // weight base += stride
    ADD                 lr8 lr8 cr13 ;;      // output base += DCS
    LDR_MULT_REG        r0 lr7 cr0 ;;        // reload R0 = weights[oc]
    BLT                 lr3 cr7 A_loop ;;    // next channel

    // ======================= Phase B: detect ============================
    LDR_MULT_REG        r0 lr0 cr14 ;;       // R0 = ones (PB)
    SET                 lr5 cr14 ;;
    ADD                 lr5 lr5 cr5 ;;       // lr5 = PB+512 (-inf plane)
    LDR_CYCLIC_MULT_REG lr5 cr0 lr0 ;;
    MULT.EE             r0 lr0 0 lr0 ;;
    ACC.FIRST ;;
    AGG.FIRST           max value lr10 cr0 aaq0 ;;   // aaq0 = -inf
    SET                 lr1 cr0 ;;           // tile offset = 0
    SET                 lr2 cr0 ;;           // tile counter = 0

B_loop:
    // chan0 -> R_ACC
    LDR_CYCLIC_MULT_REG lr1 cr3 lr0 ;;       // R_CYCLIC = chan0 tile (OUTB + tile)
    MULT.EE             r0 lr0 0 lr0 ;;
    ACC.FIRST ;;
    // chan1..3 (unrolled), addr = OUTB + c*DCS + tile
    ADD lr6 lr1 cr13 ; LDR_CYCLIC_MULT_REG lr6 cr3 lr0 ; MULT.EE r0 lr0 0 lr0 ; ACC.MAX aaq0 ;;
    ADD lr6 lr6 cr13 ; LDR_CYCLIC_MULT_REG lr6 cr3 lr0 ; MULT.EE r0 lr0 0 lr0 ; ACC.MAX aaq0 ;;
    ADD lr6 lr6 cr13 ; LDR_CYCLIC_MULT_REG lr6 cr3 lr0 ; MULT.EE r0 lr0 0 lr0 ; ACC.MAX aaq0 ;;
    // threshold: keep = relu(conf - tau)
    SET lr5 cr14 ;; ADD lr5 lr5 cr5 ;; ADD lr5 lr5 cr5 ;;   // lr5 = PB+1024 (-tau)
    LDR_CYCLIC_MULT_REG lr5 cr0 lr0 ;;
    MULT.EE             r0 lr0 0 lr0 ;;
    ACC ;;                                   // R_ACC = conf - tau
    ACTIVATE            lr10 relu ;;
    STR_POST_AAQ_REG    lr1 cr9 ;;           // keep-map at WB + tile*512 (weights now unused)
    ADD                 lr1 lr1 cr5 ;;       // next tile
    ADD                 lr2 lr2 cr1 ;;       // tile counter += 1
    BLT                 lr2 cr6 B_loop ;;

    BKPT ;;
