// ============================================================================
// cell_nms.asm  --  channel-space keypoint detection (NMS alternative)
//                   WITHOUT depth-to-space
// ----------------------------------------------------------------------------
// SuperPoint detector decision, done in CHANNEL SPACE. One cell's 64 spatial
// channels (the 8x8 sub-grid, post-softmax, dustbin already dropped) live in
// 64 lanes of one vector. Because depth-to-space only RELABELS channel c as
// sub-pixel (a,b)=(c//8, c%8), "find the local max in the cell's 8x8 block" is
// just a reduction over the channel (lane) axis -- no reshape, no spatial gather.
//
// Per cell this kernel produces three scalars:
//     peak = max_c p_c                                  -> aaq0   (confidence)
//     a    = sum_c softmax(T*p)_c * (c // 8)            -> aaq2   (row sub-coord)
//     b    = sum_c softmax(T*p)_c * (c %  8)            -> aaq3   (col sub-coord)
// With temperature T large, softmax(T*p) -> one-hot at argmax, so (a,b) is the
// (soft-)argmax sub-pixel location -- recovered as VALUES via a coordinate dot,
// so no lane-index extraction is needed. The absolute keypoint is
// (h*8 + a, w*8 + b) for cell (h,w).
//
// Numeric mode: WIDE-VECTOR FP32.  T is baked as the integer 64 via MULT.VE.CR.
//
// Register / memory contract:
//   CR0  = 0                       (zero base)
//   CR2  = scores_addr             64xf32 per-cell channel scores p (lanes 0..63)
//   CR3  = scratch_e_addr          512-byte scratch (exp values)
//   CR4  = ones_addr               128xf32 all-1.0
//   CR5  = scratch_s_addr          512-byte scratch (softmax values)
//   CR6  = rcoord_addr             128xf32: lane c -> floor(c/8) for c<64 else 0
//   CR7  = ccoord_addr             128xf32: lane c -> (c mod 8) for c<64 else 0
//   CR8  = float32_bits(-1.0)      = 0xBF800000   (negate-the-max)
//   CR9  = 64                      low byte = T (temperature) for MULT.VE.CR
//   CR10 = float32_bits(1/64)      = 0x3C800000   (rescale peak: max(T*p)/T)
//   CR11 = 64                      valid lane count N (channels per cell)
//   CR15 = dtype sentinel (INT8)
//
// Outputs (AAQ scalars): aaq0 = peak confidence, aaq2 = row sub-coord a,
//                        aaq3 = col sub-coord b.
// ============================================================================

    SET                 lr11 cr11 ;;        // N = 64 valid channels
    SET                 lr0 cr0 ;;          // 0

    // ---- xs = T * p  (T = 64 via MULT.VE.CR low byte) --------------------
    LDR_CYCLIC_MULT_REG lr0 cr2 lr0 ;;      // R_CYCLIC = p
    MULT.VE.CR          lr0 0 lr0 cr9 ;;     // MULT_RES = 64 * p
    ACC.FIRST ;;                            // R_ACC = xs = 64*p

    // ---- peak = max(p) = max(xs) * (1/64) --------------------------------
    AGG.FIRST           max value_cr lr11 cr10 aaq0 ;;  // aaq0 = peak

    // ---- one-hot: e = 2^(xs - max(xs)) -----------------------------------
    AGG.FIRST           max value_cr lr11 cr8 aaq1 ;;   // aaq1 = -max(xs)
    ACC.ADD_AAQ.FIRST   aaq1 ;;             // R_ACC = xs - max(xs)
    ACTIVATE            lr11 exp2 ;;         // POST_AAQ = e

    // ---- softmax s = e / sum(e) ------------------------------------------
    STR_POST_AAQ_REG    lr0 cr3 ;;          // scratch_e = e
    LDR_CYCLIC_MULT_REG lr0 cr4 lr0 ;;      // R_CYCLIC = ones
    LDR_MULT_REG        r0 lr0 cr3 ;;       // R0 = e
    MULT.EE             r0 lr0 0 lr0 ;;      // MULT_RES = e
    ACC.FIRST ;;                            // R_ACC = e
    AGG.FIRST           sum inv lr11 cr0 aaq1 ;;   // aaq1 = 1/sum(e)
    LDR_CYCLIC_MULT_REG lr0 cr3 lr0 ;;      // R_CYCLIC = e
    MULT.VE.AAQ         lr0 0 lr0 aaq1 ;;    // MULT_RES = e/sum(e) = s
    ACC.FIRST ;;                            // R_ACC = s
    STR_ACC_REG         lr0 cr5 ;;          // scratch_s = s

    // ---- a = sum_c s_c * rcoord_c ----------------------------------------
    LDR_MULT_REG        r0 lr0 cr5 ;;       // R0 = s
    LDR_CYCLIC_MULT_REG lr0 cr6 lr0 ;;      // R_CYCLIC = rcoord
    MULT.EE             r0 lr0 0 lr0 ;;      // MULT_RES = s * rcoord
    ACC.FIRST ;;                            // R_ACC = s*rcoord
    AGG.FIRST           sum value lr11 cr0 aaq2 ;;  // aaq2 = a (row sub-coord)

    // ---- b = sum_c s_c * ccoord_c ----------------------------------------
    LDR_MULT_REG        r0 lr0 cr5 ;;       // R0 = s
    LDR_CYCLIC_MULT_REG lr0 cr7 lr0 ;;      // R_CYCLIC = ccoord
    MULT.EE             r0 lr0 0 lr0 ;;      // MULT_RES = s * ccoord
    ACC.FIRST ;;                            // R_ACC = s*ccoord
    AGG.FIRST           sum value lr11 cr0 aaq3 ;;  // aaq3 = b (col sub-coord)

    BKPT ;;
