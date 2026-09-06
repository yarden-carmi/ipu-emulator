// ============================================================================
// l2_normalize.asm  --  L2 (unit-length) normalization over D <= 128 lanes
// ----------------------------------------------------------------------------
// Layer covered:
//   * SuperPoint descriptor head: per-keypoint descriptor L2 normalization
//     (y = x / ||x||_2).  Also the descriptor normalization SuperGlue applies
//     to its input descriptors.
//
// Numeric mode: WIDE-VECTOR FP32.
//
// Method: ||x|| = sqrt(sum x_i^2). The ISA has no sqrt post-fn but provides
//   inv_sqrt, so we compute 1/||x|| directly and scale.
//     MULT.EE.RR  -> per-lane squares
//     AGG sum inv_sqrt -> aaq0 = 1 / sqrt(sum squares) = 1/||x||
//     MULT.VE.AAQ -> x_i * (1/||x||)
//
// Register / memory contract:
//   CR0  = 0                  (zero base)
//   CR2  = vec_addr           Dxf32 input vector
//   CR3  = out_addr           Dxf32 normalized output
//   CR6  = D                  number of valid lanes (1..128)
//   CR15 = dtype sentinel (INT8)
//
// Output: out_addr[0..D) = x / ||x||_2.  If ||x|| == 0, inv_sqrt yields 0
//   (output is all zeros) by the emulator's guarded reciprocal.
// ============================================================================

    SET                 lr6 cr6 ;;          // lr6 = D
    SET                 lr0 cr0 ;;          // lr0 = 0

    LDR_MULT_REG        r0 lr0 cr2 ;;       // R0       = x
    MULT.EE.RR          r0 0 lr0 ;;         // MULT_RES = x_i * x_i
    ACC.FIRST ;;                            // R_ACC    = x^2

    AGG.FIRST           sum inv_sqrt lr6 cr0 aaq0 ;;  // aaq0 = 1 / ||x||

    LDR_CYCLIC_MULT_REG lr0 cr2 lr0 ;;      // R_CYCLIC = x
    MULT.VE.AAQ         lr0 0 lr0 aaq0 ;;    // MULT_RES = x_i * (1/||x||)
    ACC.FIRST ;;                            // R_ACC    = normalized vector

    STR_ACC_REG         lr0 cr3 ;;
    BKPT ;;
