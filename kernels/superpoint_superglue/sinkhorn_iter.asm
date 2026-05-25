// ============================================================================
// sinkhorn_iter.asm  --  one ROW half-step of LOG-domain Sinkhorn (max variant)
// ----------------------------------------------------------------------------
// Layer covered:
//   * SuperGlue optimal-matching layer: the iterative log-domain Sinkhorn
//     normalization of the (dustbin-augmented) log-score matrix Z toward a
//     doubly-stochastic assignment.
//
// LOGSUMEXP -> MAX:
//   The exact log-domain row update subtracts the row log-sum-exp:
//       Z_ij <- Z_ij - logsumexp_j(Z_ij).
//   The ISA has no logarithm, so this kernel uses the tropical (max-plus)
//   surrogate logsumexp_j(Z) ~ max_j(Z):
//       Z_ij <- Z_ij - max_j(Z_ij).
//   After the step each row's maximum is 0 and all other entries are <= 0
//   (i.e. exp(row) has its peak normalized to 1). This is the hard/tropical
//   Sinkhorn limit; it needs no reciprocal and no positivity assumption on Z.
//
// A full Sinkhorn step alternates this with the COLUMN half-step
// (Z_ij <- Z_ij - max_i Z_ij). Columns are not contiguous under the
// contiguous-only LDR, so the host transposes Z (or stores Z^T) between
// half-steps and re-calls this same kernel. See README.
//
// Layout: Z is row-major in XMEM; each row is a contiguous 128-lane (512-byte)
//   plane.  Row r is at  mat_addr + r * row_stride.  cols <= 128 (incl. dustbin).
//
// Numeric mode: WIDE-VECTOR FP32 (Z holds log-scores, any sign).
//
// Per row:  build Z[row] in R_ACC ; AGG max value_cr(-1) -> aaq0 = -max(row) ;
//           ACC.ADD_AAQ.FIRST -> Z[row] - max(row) ; store back in place.
//
// Register / memory contract:
//   CR0  = 0                  (zero base)
//   CR1  = 1                  (one; loop increment)
//   CR2  = mat_addr           base of row-major log-score matrix Z (in place)
//   CR4  = ones_addr          128xf32 all-1.0  (for the row -> R_ACC copy)
//   CR6  = num_rows           number of rows R
//   CR7  = row_stride         byte stride between rows (e.g. 512)
//   CR8  = num_cols           valid columns per row (1..128)
//   CR9  = float32_bits(-1.0) = 0xBF800000  (negate-the-max constant)
//   CR15 = dtype sentinel (INT8)
//
// Output: Z with each row shifted so its row-max is 0 (written back in place).
// ============================================================================

    SET                 lr0 cr0 ;;          // lr0 = 0
    SET                 lr6 cr6 ;;          // lr6 = num_rows
    SET                 lr8 cr8 ;;          // lr8 = num_cols
    SET                 lr5 cr0 ;;          // lr5 = 0  (row counter)
    SET                 lr3 cr2 ;;          // lr3 = current row address

    LDR_CYCLIC_MULT_REG lr0 cr4 lr0 ;;      // R_CYCLIC = ones (for row->R_ACC copy)

row_loop:
    // ---- bring Z[row] into R_ACC (MULT_RES retains Z[row]) ---------------
    LDR_MULT_REG        r0 lr3 cr0 ;;       // R0       = Z[row]   (addr = lr3)
    MULT.EE             r0 lr0 0 lr0 ;;      // MULT_RES = Z[row] * 1.0
    ACC.FIRST ;;                            // R_ACC    = Z[row]

    // ---- aaq0 = -max_j Z[row,j] ------------------------------------------
    AGG.FIRST           max value_cr lr8 cr9 aaq0 ;;   // max * (-1) = -max

    // ---- Z[row] - max(row)  (MULT_RES still = Z[row]) --------------------
    ACC.ADD_AAQ.FIRST   aaq0 ;;             // R_ACC = Z[row] + (-max)
    STR_ACC_REG         lr3 cr0 ;;          // write back at row address

    ADD                 lr3 lr3 cr7 ;;      // next row
    ADD                 lr5 lr5 cr1 ;;      // row counter += 1
    BLT                 lr5 lr6 row_loop ;;

    BKPT ;;
