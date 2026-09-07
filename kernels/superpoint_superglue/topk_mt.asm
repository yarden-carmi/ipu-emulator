// ============================================================================
// topk_mt.asm  --  multi-tile top-k support: global max + count-above-threshold
// ----------------------------------------------------------------------------
// Top-k keypoint selection over C candidate scores (C>128, T=ceil(C/128) tiles).
// The ISA cannot sort or extract ranked indices, but it CAN compute the two
// primitives that drive a threshold-calibrated top-k (the host adjusts tau):
//
//     aaq0 = max_c x_c                          (global top-1, scaled units)
//     aaq1 = sum_c sigmoid(x_c - tau)           (soft count of survivors > tau)
//
// The caller pre-scales scores by sharpness T (input = T*x) and supplies a
// tau-plane filled with T*tau; sigmoid(T*(x-tau)) -> 1 if x>tau else 0, so the
// sum is the number of candidates above tau. The host binary-searches tau until
// the count == k, then the survivors {x>tau} are the top-k set (the threshold
// mask is relu(x-tau)). Ranked order / indices remain host.
//
// Numeric mode: WIDE-VECTOR FP32.  Pad the last tile's unused lanes very
// negative so sigmoid->0 (they never count) and max ignores them.
//
// Register / memory contract:
//   CR0 = 0, CR1 = 1
//   CR2  = in_addr            T*x scores, T tiles
//   CR4  = ones_addr          128xf32 all-1.0
//   CR5  = tau_plane_addr      128xf32 all = T*tau
//   CR6  = scratch_sig_addr    512-byte (per-tile sigmoid)
//   CR7  = scratch_run_addr    512-byte (running count accumulator)
//   CR8  = num_tiles T
//   CR9  = float32_bits(-1.0)
//   CR10 = tile_stride (=512)
//   CR11 = 128
//   CR15 = dtype sentinel (INT8)
//
// Outputs: aaq0 = global max (top-1, in T*x units), aaq1 = soft count > tau.
// ============================================================================

    SET                 lr11 cr11 ;;
    SET                 lr8 cr8 ;;
    SET                 lr0 cr0 ;;
    LDR_CYCLIC_MULT_REG lr0 cr4 lr0 ;;      // R_CYCLIC = ones

    // ---- aaq2 = -(T*tau) -------------------------------------------------
    LDR_MULT_REG        r0 lr0 cr5 ;;       // R0 = tau-plane (= T*tau)
    MULT.EE             r0 lr0 0 lr0 ;;
    ACC.FIRST ;;
    AGG.FIRST           max value_cr lr11 cr9 aaq2 ;;   // aaq2 = -T*tau

    // ---- aaq0 = global max over tiles (top-1) ----------------------------
    SET                 lr4 cr2 ;;
    LDR_MULT_REG        r0 lr4 cr0 ;;
    MULT.EE             r0 lr0 0 lr0 ;;
    ACC.FIRST ;;
    AGG.FIRST           max value lr11 cr0 aaq0 ;;
    ADD                 lr4 lr4 cr10 ;;
    SET                 lr7 cr1 ;;
tk_max_loop:
    LDR_MULT_REG        r0 lr4 cr0 ;;
    MULT.EE             r0 lr0 0 lr0 ;;
    ACC.FIRST ;;
    AGG                 max value lr11 cr0 aaq0 ;;
    ADD                 lr4 lr4 cr10 ;;
    ADD                 lr7 lr7 cr1 ;;
    BLT                 lr7 lr8 tk_max_loop ;;

    // ---- running count accumulator = 0 -----------------------------------
    RESET_ACC ;;
    STR_ACC_REG         lr0 cr7 ;;          // run = 0

    // ---- count pass: run += sigmoid(T*x - T*tau) per tile ----------------
    SET                 lr4 cr2 ;;
    SET                 lr7 cr0 ;;
tk_cnt_loop:
    LDR_MULT_REG        r0 lr4 cr0 ;;       // R0 = T*x tile
    MULT.EE             r0 lr0 0 lr0 ;;      // m  = T*x
    ACC.ADD_AAQ.FIRST   aaq2 ;;             // R_ACC = T*x - T*tau
    ACTIVATE            lr11 sigmoid ;;      // POST_AAQ = sigmoid (survivor indicator)
    STR_POST_AAQ_REG    lr0 cr6 ;;          // scratch_sig = indicator
    LDR_MULT_REG        r0 lr0 cr7 ;;       // R0 = running
    MULT.EE             r0 lr0 0 lr0 ;;
    ACC.FIRST ;;                            // R_ACC = running
    LDR_MULT_REG        r0 lr0 cr6 ;;       // R0 = indicator
    MULT.EE             r0 lr0 0 lr0 ;;
    ACC ;;                                  // R_ACC = running + indicator
    STR_ACC_REG         lr0 cr7 ;;          // running updated
    ADD                 lr4 lr4 cr10 ;;
    ADD                 lr7 lr7 cr1 ;;
    BLT                 lr7 lr8 tk_cnt_loop ;;

    // ---- aaq1 = total count = sum over lanes of running ------------------
    LDR_MULT_REG        r0 lr0 cr7 ;;
    MULT.EE             r0 lr0 0 lr0 ;;
    ACC.FIRST ;;
    AGG.FIRST           sum value lr11 cr0 aaq1 ;;

    BKPT ;;
