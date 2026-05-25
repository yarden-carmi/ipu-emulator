// ============================================================================
// sinkhorn_col.asm  --  COLUMN half-step of log-domain Sinkhorn (max variant)
//                       without any matrix transpose
// ----------------------------------------------------------------------------
// Companion to sinkhorn_iter.asm (the row half-step). Together they form one
// full log-domain Sinkhorn iteration entirely on-device:
//       row:  Z_ij <- Z_ij - max_j Z_ij      (sinkhorn_iter.asm)
//       col:  Z_ij <- Z_ij - max_i Z_ij      (THIS kernel)
//
// THE TRICK (no transpose / no column gather):
//   max_i Z_ij -- the per-COLUMN maximum -- is the ELEMENT-WISE maximum of the
//   row vectors. Rows are contiguous, so we sweep them with a running
//   element-wise max (ACC.MAX) to build a single colmax vector, then subtract
//   that one vector from every row. Both sweeps read only contiguous 128-lane
//   row planes -- the strided column access that blocked the column step is
//   avoided entirely.
//
//   Pass 1 (accumulate):   colmax_j = max_i Z_ij        (ACC.MAX across rows)
//   negate:                negcol_j = -colmax_j
//   Pass 2 (apply):        Z_ij <- Z_ij + negcol_j = Z_ij - max_i Z_ij
//
// Numeric mode: WIDE-VECTOR FP32 (Z holds log-scores, any sign).
//
// Register / memory contract:
//   CR0  = 0                  (zero base)
//   CR1  = 1                  (one; loop increment)
//   CR2  = mat_addr           base of row-major log-score matrix Z (in place)
//   CR4  = ones_addr          128xf32 all-1.0   (for x1 vector copies)
//   CR5  = neg_ones_addr      128xf32 all -1.0  (for vector negation)
//   CR6  = num_rows           number of rows R
//   CR7  = row_stride         byte stride between rows (e.g. 512)
//   CR8  = num_cols           valid columns (1..128); used only to seed -inf
//   CR9  = colmax_addr        512-byte scratch for the column-max vector
//   CR10 = negcol_addr        512-byte scratch for -(column-max) vector
//   CR11 = neginf_addr        128xf32 all -3.4e38  (ACC.MAX seed)
//   CR15 = dtype sentinel (INT8)
//
// Output: Z with each column shifted so its column-max is 0 (in place).
// ============================================================================

    SET                 lr0 cr0 ;;          // lr0 = 0
    SET                 lr6 cr6 ;;          // lr6 = num_rows
    SET                 lr8 cr8 ;;          // lr8 = num_cols

    LDR_CYCLIC_MULT_REG lr0 cr4 lr0 ;;      // R_CYCLIC = ones

    // ---- seed aaq0 = -inf  (neutral element for the 3-way ACC.MAX) -------
    LDR_MULT_REG        r0 lr0 cr11 ;;      // R0  = -inf plane
    MULT.EE             r0 lr0 0 lr0 ;;      // m   = -inf
    ACC.FIRST ;;                            // a   = -inf
    AGG.FIRST           max value lr8 cr0 aaq0 ;;   // aaq0 = -inf

    // ---- Pass 1: colmax = element-wise max over all rows -----------------
    SET                 lr3 cr2 ;;          // lr3 = row address
    LDR_MULT_REG        r0 lr3 cr0 ;;       // R0  = Z[row0]
    MULT.EE             r0 lr0 0 lr0 ;;      // m   = Z[row0]
    ACC.FIRST ;;                            // a   = Z[row0]   (running colmax)
    SET                 lr5 cr1 ;;          // lr5 = 1  (row counter)
    ADD                 lr3 lr3 cr7 ;;      // -> row1
colmax_loop:
    LDR_MULT_REG        r0 lr3 cr0 ;;       // R0  = Z[row]
    MULT.EE             r0 lr0 0 lr0 ;;      // m   = Z[row]
    ACC.MAX             aaq0 ;;             // a   = max(a, Z[row], -inf)
    ADD                 lr3 lr3 cr7 ;;      // next row
    ADD                 lr5 lr5 cr1 ;;      // counter += 1
    BLT                 lr5 lr6 colmax_loop ;;
    STR_ACC_REG         lr0 cr9 ;;          // colmax_addr = colmax

    // ---- negate: negcol = -colmax ----------------------------------------
    LDR_CYCLIC_MULT_REG lr0 cr5 lr0 ;;      // R_CYCLIC = -1.0
    LDR_MULT_REG        r0 lr0 cr9 ;;       // R0  = colmax
    MULT.EE             r0 lr0 0 lr0 ;;      // m   = colmax * (-1)
    ACC.FIRST ;;                            // a   = -colmax
    STR_ACC_REG         lr0 cr10 ;;         // negcol_addr = -colmax

    // ---- Pass 2: Z[row] = Z[row] - colmax  for every row -----------------
    LDR_CYCLIC_MULT_REG lr0 cr4 lr0 ;;      // R_CYCLIC = ones
    SET                 lr3 cr2 ;;          // lr3 = row address
    SET                 lr5 cr0 ;;          // lr5 = 0  (row counter)
apply_loop:
    LDR_MULT_REG        r0 lr3 cr0 ;;       // R0  = Z[row]
    MULT.EE             r0 lr0 0 lr0 ;;      // m   = Z[row]
    ACC.FIRST ;;                            // a   = Z[row]
    LDR_MULT_REG        r0 lr0 cr10 ;;      // R0  = -colmax  (addr = 0 + cr10)
    MULT.EE             r0 lr0 0 lr0 ;;      // m   = -colmax
    ACC ;;                                  // a   = Z[row] + (-colmax)
    STR_ACC_REG         lr3 cr0 ;;          // write back at row address
    ADD                 lr3 lr3 cr7 ;;      // next row
    ADD                 lr5 lr5 cr1 ;;      // counter += 1
    BLT                 lr5 lr6 apply_loop ;;

    BKPT ;;
