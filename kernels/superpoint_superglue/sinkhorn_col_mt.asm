// ============================================================================
// sinkhorn_col_mt.asm  --  COLUMN half-step of log-domain Sinkhorn (max),
//                          MULTI-TILE, transpose-free
// ----------------------------------------------------------------------------
// Generalizes sinkhorn_col.asm to C columns = T column-tiles. Performs
//       Z_ij <- Z_ij - max_i Z_ij      (column max, no transpose)
// by building, for each column-tile, the element-wise max across all rows
// (ACC.MAX over contiguous row planes), then subtracting it from every row.
//
// Address of (row i, col-tile t) = mat_addr + i*row_stride + t*tile_stride.
//
// Numeric mode: WIDE-VECTOR FP32.
//
// Register / memory contract:
//   CR0  = 0                  (zero base)
//   CR1  = 1                  (one; loop increment)
//   CR2  = mat_addr           base of row-major matrix Z (in place)
//   CR4  = ones_addr          128xf32 all-1.0
//   CR5  = neg_ones_addr      128xf32 all -1.0
//   CR6  = num_rows           R
//   CR7  = row_stride         bytes between rows (= T*512 packed)
//   CR8  = num_col_tiles      T
//   CR10 = tile_stride        bytes between col-tiles (= 512)
//   CR11 = 128                lane count for the -inf seed reduction
//   CR12 = neginf_addr        128xf32 all -3.4e38
//   CR13 = colmax_base        T*512 scratch (column-max tiles)
//   CR14 = negcol_base        T*512 scratch (-(column-max) tiles)
//   CR15 = dtype sentinel (INT8)
//
// Output: every column shifted so its column-max is 0 (in place).
// ============================================================================

    SET                 lr0 cr0 ;;          // 0
    SET                 lr6 cr6 ;;          // R
    SET                 lr8 cr8 ;;          // T
    SET                 lr11 cr11 ;;        // 128
    LDR_CYCLIC_MULT_REG lr0 cr4 lr0 ;;      // R_CYCLIC = ones

    // ---- seed aaq0 = -inf (neutral for 3-way ACC.MAX) --------------------
    LDR_MULT_REG        r0 lr0 cr12 ;;      // R0 = -inf plane
    MULT.EE             r0 lr0 0 lr0 ;;      // m  = -inf
    ACC.FIRST ;;                            // a  = -inf
    AGG.FIRST           max value lr11 cr0 aaq0 ;;  // aaq0 = -inf

    // ================= Pass 1: colmax per col-tile ========================
    SET                 lr9 cr2 ;;          // lr9 = col-tile base in Z (ct=0)
    SET                 lr10 cr13 ;;        // lr10 = colmax scratch slot
    SET                 lr7 cr0 ;;          // ct counter = 0
coltile_loop:
    ADD                 lr4 lr9 cr0 ;;      // lr4 = row addr = ct base (row0)
    LDR_MULT_REG        r0 lr4 cr0 ;;       // R0 = Z[row0, ct]
    MULT.EE             r0 lr0 0 lr0 ;;      // m  = Z[row0, ct]
    ACC.FIRST ;;                            // a  = Z[row0, ct]  (running colmax)
    SET                 lr5 cr1 ;;          // row counter = 1
    ADD                 lr4 lr4 cr7 ;;      // -> row1
cm_row_loop:
    LDR_MULT_REG        r0 lr4 cr0 ;;       // R0 = Z[row, ct]
    MULT.EE             r0 lr0 0 lr0 ;;      // m  = Z[row, ct]
    ACC.MAX             aaq0 ;;             // a  = max(a, Z[row,ct], -inf)
    ADD                 lr4 lr4 cr7 ;;      // next row
    ADD                 lr5 lr5 cr1 ;;
    BLT                 lr5 lr6 cm_row_loop ;;
    STR_ACC_REG         lr10 cr0 ;;         // colmax[ct] stored
    ADD                 lr9 lr9 cr10 ;;     // next col-tile in Z
    ADD                 lr10 lr10 cr10 ;;   // next colmax slot
    ADD                 lr7 lr7 cr1 ;;
    BLT                 lr7 lr8 coltile_loop ;;

    // ================= negate: negcol[ct] = -colmax[ct] ===================
    LDR_CYCLIC_MULT_REG lr0 cr5 lr0 ;;      // R_CYCLIC = -1.0
    SET                 lr9 cr13 ;;         // colmax slot
    SET                 lr10 cr14 ;;        // negcol slot
    SET                 lr7 cr0 ;;          // ct counter = 0
neg_loop:
    LDR_MULT_REG        r0 lr9 cr0 ;;       // R0 = colmax[ct]
    MULT.EE             r0 lr0 0 lr0 ;;      // m  = colmax * (-1)
    ACC.FIRST ;;                            // a  = -colmax
    STR_ACC_REG         lr10 cr0 ;;         // negcol[ct]
    ADD                 lr9 lr9 cr10 ;;
    ADD                 lr10 lr10 cr10 ;;
    ADD                 lr7 lr7 cr1 ;;
    BLT                 lr7 lr8 neg_loop ;;

    // ================= Pass 2: Z[row,ct] += negcol[ct] ====================
    LDR_CYCLIC_MULT_REG lr0 cr4 lr0 ;;      // R_CYCLIC = ones
    SET                 lr3 cr2 ;;          // row base in Z
    SET                 lr5 cr0 ;;          // row counter = 0
apply_row_loop:
    ADD                 lr4 lr3 cr0 ;;      // lr4 = tile addr = row base
    SET                 lr9 cr14 ;;         // negcol slot (ct=0)
    SET                 lr7 cr0 ;;          // ct counter = 0
apply_tile_loop:
    LDR_MULT_REG        r0 lr4 cr0 ;;       // R0 = Z[row, ct]
    MULT.EE             r0 lr0 0 lr0 ;;      // m  = Z[row, ct]
    ACC.FIRST ;;                            // a  = Z[row, ct]
    LDR_MULT_REG        r0 lr9 cr0 ;;       // R0 = negcol[ct]
    MULT.EE             r0 lr0 0 lr0 ;;      // m  = negcol[ct]
    ACC ;;                                  // a  = Z[row,ct] - colmax[ct]
    STR_ACC_REG         lr4 cr0 ;;          // write back
    ADD                 lr4 lr4 cr10 ;;     // next col-tile in Z
    ADD                 lr9 lr9 cr10 ;;     // next negcol slot
    ADD                 lr7 lr7 cr1 ;;
    BLT                 lr7 lr8 apply_tile_loop ;;
    ADD                 lr3 lr3 cr7 ;;      // next row
    ADD                 lr5 lr5 cr1 ;;
    BLT                 lr5 lr6 apply_row_loop ;;

    BKPT ;;
