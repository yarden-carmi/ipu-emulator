// ============================================================================
// sinkhorn_iter.asm  --  one ROW-normalization half-step of Sinkhorn matching
// ----------------------------------------------------------------------------
// Layer covered:
//   * SuperGlue optimal-matching layer: the iterative Sinkhorn normalization
//     of the (augmented, exponentiated) score matrix toward a doubly-stochastic
//     assignment.
//
// DOMAIN CHOICE (important): the standard log-domain Sinkhorn uses logsumexp,
//   which needs a logarithm. The ISA exposes exp2 but NO log, so this kernel
//   implements the *standard-domain* iteration on the positive affinity matrix
//   P (P_ij >= 0, e.g. P = 2^score):
//       row half-step:  P_ij <- P_ij / sum_j P_ij      (each row sums to 1)
//   A full Sinkhorn step alternates this with the analogous COLUMN
//   normalization. The column half-step needs transposed/strided column loads
//   the contiguous-only LDR cannot gather; the host transposes P between
//   half-steps (or stores P^T) and calls this same kernel. See README.
//
// Layout: P is row-major in XMEM; each row is a contiguous 128-lane (512-byte)
//   plane.  Row r is at  mat_addr + r * row_stride.  cols <= 128 (incl. dustbin).
//
// Numeric mode: WIDE-VECTOR FP32.
//
// Per row:  build row in R_ACC ; AGG sum inv -> aaq0 = 1/rowsum ;
//           MULT.VE.AAQ scales the row ; store back in place.
//
// Register / memory contract:
//   CR0  = 0                  (zero base)
//   CR1  = 1                  (one; loop increment)
//   CR2  = mat_addr           base of row-major matrix P (output in place)
//   CR4  = ones_addr          128xf32 all-1.0
//   CR6  = num_rows           number of rows R
//   CR7  = row_stride         byte stride between rows (e.g. 512)
//   CR8  = num_cols           valid columns per row (1..128)
//   CR15 = dtype sentinel (INT8)
//
// Output: P normalized so every row sums to 1 (written back in place).
// ============================================================================

    SET                 lr0 cr0 ;;          // lr0 = 0
    SET                 lr6 cr6 ;;          // lr6 = num_rows
    SET                 lr8 cr8 ;;          // lr8 = num_cols
    SET                 lr5 cr0 ;;          // lr5 = 0  (row counter)
    SET                 lr3 cr2 ;;          // lr3 = current row address

    LDR_CYCLIC_MULT_REG lr0 cr4 lr0 ;;      // R_CYCLIC = ones (for row->R_ACC copy)

row_loop:
    // ---- bring P[row] into R_ACC -----------------------------------------
    LDR_MULT_REG        r0 lr3 cr0 ;;       // R0       = P[row]   (addr = lr3)
    MULT.EE             r0 lr0 0 lr0 ;;      // MULT_RES = P[row] * 1.0
    ACC.FIRST ;;                            // R_ACC    = P[row]

    // ---- aaq0 = 1 / sum_j P[row,j] ---------------------------------------
    AGG.FIRST           sum inv lr8 cr0 aaq0 ;;

    // ---- scale row by 1/rowsum and store in place ------------------------
    LDR_CYCLIC_MULT_REG lr3 cr0 lr0 ;;      // R_CYCLIC = P[row]
    MULT.VE.AAQ         lr0 0 lr0 aaq0 ;;    // MULT_RES = P[row,j] / rowsum
    ACC.FIRST ;;                            // R_ACC    = normalized row
    STR_ACC_REG         lr3 cr0 ;;          // write back at row address

    // restore ones in R_CYCLIC for the next iteration's copy
    LDR_CYCLIC_MULT_REG lr0 cr4 lr0 ;;      // R_CYCLIC = ones

    ADD                 lr3 lr3 cr7 ;;      // next row
    ADD                 lr5 lr5 cr1 ;;      // row counter += 1
    BLT                 lr5 lr6 row_loop ;;

    BKPT ;;
