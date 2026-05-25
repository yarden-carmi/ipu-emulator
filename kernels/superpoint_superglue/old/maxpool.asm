// ============================================================================
// maxpool.asm  --  element-wise max over K "taps" (spatial max-pool / NMS core)
// ----------------------------------------------------------------------------
// Layer covered:
//   * SuperPoint keypoint post-processing: max-pooling and the local-maximum
//     test used by non-maximum suppression (NMS).
//
// Model: the K window positions that map to one output lane are pre-gathered by
//   the caller into K contiguous 128-lane (512-byte) "tap" planes in XMEM:
//       tap_t at address  in_addr + t * tap_stride   (t = 0..K-1)
//   The kernel computes, per lane i:  out[i] = max_t tap_t[i].
//   For a 2x2 pool, K=4 and each tap plane is one of the 4 sub-sampled grids;
//   for NMS over a (2r+1)x(2r+1) window, K = (2r+1)^2 neighbour shifts.
//
// Numeric mode: WIDE-VECTOR FP32.
//
// aaq0 is seeded to -inf so the 3-way ACC.MAX (max of R_ACC, MULT_RES, aaq0)
// reduces to the desired 2-way running max. The caller provides a -inf plane.
//
// Register / memory contract:
//   CR0  = 0                       (zero base)
//   CR1  = 1                       (one; loop increment)
//   CR2  = in_addr                 base address of tap0 plane
//   CR3  = out_addr                128xf32 pooled output
//   CR4  = ones_addr               128xf32 all-1.0
//   CR5  = neginf_addr             128xf32 all -3.4e38 (-inf seed)
//   CR6  = K                       number of taps (>=1)
//   CR7  = tap_stride              byte stride between tap planes (e.g. 512)
//   CR8  = 128                     full lane count (for the -inf reduction)
//   CR15 = dtype sentinel (INT8)
// ============================================================================

    SET                 lr0 cr0 ;;          // lr0 = 0  (offset / index)
    SET                 lr6 cr6 ;;          // lr6 = K
    SET                 lr8 cr8 ;;          // lr8 = 128
    SET                 lr5 cr1 ;;          // lr5 = 1  (tap counter; tap0 done below)
    SET                 lr3 cr2 ;;          // lr3 = address offset, starts at in_addr

    // ---- ones into R_CYCLIC (kept for all x1 copies) ---------------------
    LDR_CYCLIC_MULT_REG lr0 cr4 lr0 ;;      // R_CYCLIC = ones

    // ---- seed aaq0 = -inf -------------------------------------------------
    LDR_MULT_REG        r0 lr0 cr5 ;;       // R0       = -inf plane
    MULT.EE             r0 lr0 0 lr0 ;;      // MULT_RES = -inf
    ACC.FIRST ;;                            // R_ACC    = -inf
    AGG.FIRST           max value lr8 cr0 aaq0 ;;   // aaq0 = -inf

    // ---- tap0 -> R_ACC ----------------------------------------------------
    LDR_MULT_REG        r0 lr0 cr2 ;;       // R0       = tap0
    MULT.EE             r0 lr0 0 lr0 ;;      // MULT_RES = tap0
    ACC.FIRST ;;                            // R_ACC    = tap0

    // ---- running max over taps 1..K-1 ------------------------------------
    ADD                 lr3 lr3 cr7 ;;      // lr3 = in_addr + tap_stride
pool_loop:
    // addr = lr3 (moving tap base) + CR0(=0).
    LDR_MULT_REG        r0 lr3 cr0 ;;       // R0       = tap_t  (addr = lr3)
    MULT.EE             r0 lr0 0 lr0 ;;      // MULT_RES = tap_t
    ACC.MAX             aaq0 ;;             // R_ACC = max(R_ACC, tap_t, -inf)
    ADD                 lr3 lr3 cr7 ;;      // advance to next tap plane
    ADD                 lr5 lr5 cr1 ;;      // tap counter += 1
    BLT                 lr5 lr6 pool_loop ;;

    STR_ACC_REG         lr0 cr3 ;;
    BKPT ;;
