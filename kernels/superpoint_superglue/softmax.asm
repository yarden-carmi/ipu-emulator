// ============================================================================
// softmax.asm  --  numerically-stable softmax over N <= 128 lanes
// ----------------------------------------------------------------------------
// Layers covered:
//   * SuperPoint detector head: per-cell channel softmax (65 bins).
//   * SuperGlue attention: row softmax over attention logits.
//
// Numeric mode: WIDE-VECTOR FP32 (IpuState(wide_vector_debug=True,
//               wide_vector_arithmetic=FP32)). All 128 lanes are float32.
//
// Base-2 note: the only exponential the ISA exposes is ACTIVATE exp2 (2^x).
//   softmax_2(x_i) = 2^(x_i - m) / sum_j 2^(x_j - m).
//   To obtain natural-base softmax, the caller must pre-scale the logits by
//   log2(e) = 1.4426950408 before invoking this kernel. Stability shift by the
//   max m is applied here regardless of base.
//
// Register / memory contract (caller sets these before run):
//   CR0  = 0                      (zero base; hardwired)
//   CR1  = 1                      (one; hardwired)
//   CR2  = logits_addr            128xf32 input logits at this XMEM byte addr
//   CR3  = scratch_addr           512-byte scratch buffer (kernel-owned)
//   CR4  = ones_addr              128xf32 all-1.0 (used for x1 copies)
//   CR5  = out_addr               128xf32 output probabilities written here
//   CR6  = N                      number of valid lanes (1..128)
//   CR7  = float32_bits(-1.0)     = 0xBF800000  (negate-the-max constant)
//   CR15 = dtype sentinel (INT8)  required by wide-mode setup
//
// Output: out_addr[0..N) = softmax(logits[0..N)); lanes [N..128) undefined.
// ============================================================================

    SET                 lr6 cr6 ;;          // lr6 = N (valid lane count)
    SET                 lr0 cr0 ;;          // lr0 = 0  (offset / index 0)

    // ---- bring logits into R_ACC -----------------------------------------
    LDR_MULT_REG        r0 lr0 cr2 ;;       // R0       = logits
    LDR_CYCLIC_MULT_REG lr0 cr4 lr0 ;;      // R_CYCLIC = ones (1.0)
    MULT.EE             r0 lr0 0 lr0 ;;      // MULT_RES = logits * 1.0
    ACC.FIRST ;;                            // R_ACC    = logits

    // ---- m = max(logits); store -m in aaq0 -------------------------------
    AGG.FIRST           max value_cr lr6 cr7 aaq0 ;;  // aaq0 = max * (-1.0) = -m

    // ---- shift: R_ACC = logits - m ---------------------------------------
    // MULT_RES still holds logits from the MULT.EE above.
    ACC.ADD_AAQ.FIRST   aaq0 ;;             // R_ACC = MULT_RES + (-m) = logits - m

    // ---- exp: POST_AAQ = 2^(logits - m) ----------------------------------
    ACTIVATE            lr6 exp2 ;;          // POST_AAQ_REG[i] = 2^(R_ACC[i])

    // ---- round-trip exps through XMEM so AGG can reduce them --------------
    STR_POST_AAQ_REG    lr0 cr3 ;;          // scratch = exps (512 B)
    LDR_MULT_REG        r0 lr0 cr3 ;;       // R0       = exps
    MULT.EE             r0 lr0 0 lr0 ;;      // MULT_RES = exps * 1.0  (R_CYCLIC=ones)
    ACC.FIRST ;;                            // R_ACC    = exps

    // ---- 1 / sum(exps) ---------------------------------------------------
    AGG.FIRST           sum inv lr6 cr0 aaq0 ;;  // aaq0 = 1 / sum_j exp_j

    // ---- probs = exps * (1/sum) ------------------------------------------
    LDR_CYCLIC_MULT_REG lr0 cr3 lr0 ;;      // R_CYCLIC = exps
    MULT.VE.AAQ         lr0 0 lr0 aaq0 ;;    // MULT_RES = (1/sum) * exps[i]
    ACC.FIRST ;;                            // R_ACC    = softmax probabilities

    STR_ACC_REG         lr0 cr5 ;;          // out_addr = probabilities
    BKPT ;;
