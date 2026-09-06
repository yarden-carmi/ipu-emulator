// ============================================================================
// topk.asm  --  keypoint selection: confidence threshold + global maximum
// ----------------------------------------------------------------------------
// Layer covered:
//   * SuperPoint keypoint selection. The detector keeps points whose score
//     exceeds a confidence threshold and then keeps the strongest ones.
//
// WHAT THIS KERNEL DOES (the on-device-expressible part):
//   1. Thresholded scores:  sel_i = relu(x_i - thr)   (kept points keep a
//      positive, shifted score; rejected points become 0).
//   2. Top-1 value:         m = max_i x_i             (broadcast to a plane).
//
// WHAT IT DOES NOT DO: produce the ranked list of the k largest INDICES.
//   Exact ranked top-k needs per-element compare + index scatter, and the ISA
//   exposes neither a vector compare nor lane-index extraction (see README
//   LIMITATIONS). The host selects the final k from the thresholded scores.
//
// Numeric mode: WIDE-VECTOR FP32.
//
// Register / memory contract:
//   CR0  = 0                       (zero base)
//   CR2  = scores_addr             Nxf32 detector scores
//   CR3  = thr_addr                Nxf32 plane all equal to the threshold thr
//   CR4  = ones_addr               128xf32 all-1.0
//   CR5  = sel_addr                Nxf32 output: relu(x - thr)
//   CR6  = N                       valid lanes (1..128)
//   CR7  = float32_bits(-1.0)      = 0xBF800000
//   CR8  = max_addr                f32 plane output: broadcast top-1 value
//   CR15 = dtype sentinel (INT8)
// ============================================================================

    SET                 lr6 cr6 ;;          // lr6 = N
    SET                 lr0 cr0 ;;          // lr0 = 0

    LDR_CYCLIC_MULT_REG lr0 cr4 lr0 ;;      // R_CYCLIC = ones

    // ---- aaq0 = -thr  (max of the constant thr-plane, negated) -----------
    LDR_MULT_REG        r0 lr0 cr3 ;;       // R0       = thr-plane
    MULT.EE             r0 lr0 0 lr0 ;;      // MULT_RES = thr
    ACC.FIRST ;;                            // R_ACC    = thr
    AGG.FIRST           max value_cr lr6 cr7 aaq0 ;;   // aaq0 = -thr

    // ---- R_ACC = x - thr  then relu --------------------------------------
    LDR_MULT_REG        r0 lr0 cr2 ;;       // R0       = scores x
    MULT.EE             r0 lr0 0 lr0 ;;      // MULT_RES = x
    ACC.ADD_AAQ.FIRST   aaq0 ;;             // R_ACC    = x + (-thr) = x - thr
    ACTIVATE            lr6 relu ;;          // POST_AAQ = relu(x - thr)
    STR_POST_AAQ_REG    lr0 cr5 ;;          // sel_addr = thresholded scores

    // ---- top-1 value: m = max x ; broadcast ; store ----------------------
    MULT.EE             r0 lr0 0 lr0 ;;      // MULT_RES = x   (R0 still = x)
    ACC.FIRST ;;                            // R_ACC    = x
    AGG.FIRST           max value lr6 cr0 aaq0 ;;     // aaq0 = max x
    LDR_CYCLIC_MULT_REG lr0 cr4 lr0 ;;      // R_CYCLIC = ones
    MULT.VE.AAQ         lr0 0 lr0 aaq0 ;;    // MULT_RES = max (broadcast)
    ACC.FIRST ;;                            // R_ACC    = max in all lanes
    STR_ACC_REG         lr0 cr8 ;;          // max_addr = top-1 value
    BKPT ;;
