// ============================================================================
// layernorm.asm  --  layer normalization with affine, over N <= 128 lanes
// ----------------------------------------------------------------------------
// Layer covered:
//   * SuperGlue MLP / attention block normalization:
//       y = gamma' * (x - mu) * rsqrt(sum_i (x_i - mu)^2) + beta
//     where mu = mean(x).
//
// IMPORTANT scaling convention (read carefully):
//   The exact LayerNorm denominator is sigma = sqrt(mean((x-mu)^2)) =
//   sqrt(sum/N). This kernel divides by sqrt(SUM) (not the mean), i.e. it
//   multiplies (x-mu) by rsqrt(sum (x-mu)^2) = 1/(sqrt(N)*sigma). The missing
//   sqrt(N) factor is FOLDED INTO THE CALLER-SUPPLIED gamma':
//       gamma'_i = gamma_i * sqrt(N)
//   With that, the result equals gamma_i * (x_i - mu)/sigma + beta_i exactly.
//   (epsilon is omitted; inv_sqrt is guarded to 0 when the sum is <= 0.)
//
// Numeric mode: WIDE-VECTOR FP32.
//
// Register / memory contract:
//   CR0  = 0                       (zero base)
//   CR1  = 1                       (one)
//   CR2  = x_addr                  Nxf32 input
//   CR3  = scratch_addr            512-byte kernel scratch
//   CR4  = float32_bits(-1.0/N)    negate-and-mean constant (for -mu)
//   CR5  = out_addr                Nxf32 output
//   CR6  = N                       valid lanes (1..128)
//   CR8  = gamma_addr              Nxf32 affine scale, pre-multiplied by sqrt(N)
//   CR9  = beta_addr               Nxf32 affine bias
//   CR10 = ones_addr               Nxf32 all-1.0
//   CR15 = dtype sentinel (INT8)
// ============================================================================

    SET                 lr6 cr6 ;;          // lr6 = N
    SET                 lr0 cr0 ;;          // lr0 = 0

    // ---- x into R_ACC -----------------------------------------------------
    LDR_MULT_REG        r0 lr0 cr2 ;;       // R0       = x
    LDR_CYCLIC_MULT_REG lr0 cr10 lr0 ;;     // R_CYCLIC = ones
    MULT.EE             r0 lr0 0 lr0 ;;      // MULT_RES = x
    ACC.FIRST ;;                            // R_ACC    = x

    // ---- -mu = sum(x) * (-1/N) -> aaq0 -----------------------------------
    AGG.FIRST           sum value_cr lr6 cr4 aaq0 ;;   // aaq0 = -mu

    // ---- centered: R_ACC = x - mu (MULT_RES still = x) -------------------
    ACC.ADD_AAQ.FIRST   aaq0 ;;             // R_ACC = x + (-mu)

    // ---- square the centered values: store, reload, MULT.EE.RR -----------
    STR_ACC_REG         lr0 cr3 ;;          // scratch = (x - mu)
    LDR_MULT_REG        r0 lr0 cr3 ;;       // R0       = (x - mu)
    MULT.EE.RR          r0 0 lr0 ;;         // MULT_RES = (x - mu)^2
    ACC.FIRST ;;                            // R_ACC    = (x - mu)^2

    // ---- aaq0 = rsqrt(sum (x-mu)^2) = 1/(sqrt(N)*sigma) ------------------
    AGG.FIRST           sum inv_sqrt lr6 cr0 aaq0 ;;

    // ---- z = (x - mu) * rsqrt(sum) ---------------------------------------
    LDR_CYCLIC_MULT_REG lr0 cr3 lr0 ;;      // R_CYCLIC = (x - mu)
    MULT.VE.AAQ         lr0 0 lr0 aaq0 ;;    // MULT_RES = (x-mu)*rsqrt(sum)
    ACC.FIRST ;;                            // R_ACC    = z

    // ---- affine scale: R_ACC = z * gamma' --------------------------------
    STR_ACC_REG         lr0 cr3 ;;          // scratch = z
    LDR_MULT_REG        r0 lr0 cr3 ;;       // R0       = z
    LDR_CYCLIC_MULT_REG lr0 cr8 lr0 ;;      // R_CYCLIC = gamma' (= gamma*sqrt(N))
    MULT.EE             r0 lr0 0 lr0 ;;      // MULT_RES = z * gamma'
    ACC.FIRST ;;                            // R_ACC    = z * gamma'

    // ---- affine bias: R_ACC += beta --------------------------------------
    LDR_MULT_REG        r0 lr0 cr9 ;;       // R0       = beta
    LDR_CYCLIC_MULT_REG lr0 cr10 lr0 ;;     // R_CYCLIC = ones
    MULT.EE             r0 lr0 0 lr0 ;;      // MULT_RES = beta * 1.0
    ACC ;;                                  // R_ACC   += beta

    STR_ACC_REG         lr0 cr5 ;;
    BKPT ;;
