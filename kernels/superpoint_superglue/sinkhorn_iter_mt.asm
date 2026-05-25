// ============================================================================
// sinkhorn_iter_mt.asm  --  ROW half-step of log-domain Sinkhorn (max variant),
//                           MULTI-TILE: handles rows wider than 128 columns
// ----------------------------------------------------------------------------
// Generalizes sinkhorn_iter.asm to C columns laid out as T = ceil(C/128)
// contiguous 128-lane (512-byte) column-tiles per row. Performs, per row,
//       Z_ij <- Z_ij - max_j Z_ij      (max over ALL C columns)
//
// The row max over >128 columns is built by folding the per-tile max:
// `AGG max value` re-includes the running AAQ, so sweeping tiles with it
// accumulates the global row max. (Pad unused lanes of the last tile with
// -inf so they never win the max.)
//
// Numeric mode: WIDE-VECTOR FP32 (Z holds log-scores, any sign).
//
// Address of (row i, col-tile t) = mat_addr + i*row_stride + t*tile_stride.
//
// Register / memory contract:
//   CR0  = 0                  (zero base)
//   CR1  = 1                  (one; loop increment)
//   CR2  = mat_addr           base of row-major matrix Z (in place)
//   CR4  = ones_addr          128xf32 all-1.0
//   CR6  = num_rows           R
//   CR7  = row_stride         bytes between rows (= T*512 for a packed matrix)
//   CR8  = num_col_tiles      T = ceil(C/128)
//   CR9  = float32_bits(-1.0) = 0xBF800000
//   CR10 = tile_stride        bytes between col-tiles within a row (= 512)
//   CR11 = 128                lane count for the AGG reductions
//   CR15 = dtype sentinel (INT8)
//
// Output: every row shifted so its max over all C columns is 0 (in place).
// ============================================================================

    SET                 lr0 cr0 ;;          // 0
    SET                 lr6 cr6 ;;          // R
    SET                 lr8 cr8 ;;          // T (col-tiles)
    SET                 lr11 cr11 ;;        // 128
    LDR_CYCLIC_MULT_REG lr0 cr4 lr0 ;;      // R_CYCLIC = ones (x1 copies + broadcast)

    SET                 lr3 cr2 ;;          // lr3 = base address of current row
    SET                 lr5 cr0 ;;          // lr5 = 0  (row counter)

row_loop:
    // ---- pass A: rowmax over T tiles -> aaq0 -----------------------------
    ADD                 lr4 lr3 cr0 ;;      // lr4 = tile addr = row base
    LDR_MULT_REG        r0 lr4 cr0 ;;       // R0 = Z[row, tile0]
    MULT.EE             r0 lr0 0 lr0 ;;      // m  = tile0
    ACC.FIRST ;;                            // a  = tile0
    AGG.FIRST           max value lr11 cr0 aaq0 ;;  // aaq0 = max(tile0)
    ADD                 lr4 lr4 cr10 ;;     // -> tile1
    SET                 lr7 cr1 ;;          // tile counter = 1
tilemax_loop:
    LDR_MULT_REG        r0 lr4 cr0 ;;       // R0 = Z[row, tile_t]
    MULT.EE             r0 lr0 0 lr0 ;;      // m  = tile_t
    ACC.FIRST ;;                            // a  = tile_t
    AGG                 max value lr11 cr0 aaq0 ;;  // aaq0 = max(tile_t, aaq0)
    ADD                 lr4 lr4 cr10 ;;     // next tile
    ADD                 lr7 lr7 cr1 ;;
    BLT                 lr7 lr8 tilemax_loop ;;

    // ---- aaq1 = -rowmax --------------------------------------------------
    MULT.VE.AAQ         lr0 0 lr0 aaq0 ;;    // m = rowmax (broadcast over ones)
    ACC.FIRST ;;                            // a = rowmax
    AGG.FIRST           max value_cr lr11 cr9 aaq1 ;;  // aaq1 = rowmax*(-1)

    // ---- pass B: subtract rowmax from every tile -------------------------
    ADD                 lr4 lr3 cr0 ;;      // lr4 = tile addr = row base
    SET                 lr7 cr0 ;;          // tile counter = 0
tilesub_loop:
    LDR_MULT_REG        r0 lr4 cr0 ;;       // R0 = Z[row, tile_t]
    MULT.EE             r0 lr0 0 lr0 ;;      // m  = tile_t
    ACC.ADD_AAQ.FIRST   aaq1 ;;             // a  = tile_t + (-rowmax)
    STR_ACC_REG         lr4 cr0 ;;          // write tile back
    ADD                 lr4 lr4 cr10 ;;
    ADD                 lr7 lr7 cr1 ;;
    BLT                 lr7 lr8 tilesub_loop ;;

    ADD                 lr3 lr3 cr7 ;;      // next row
    ADD                 lr5 lr5 cr1 ;;
    BLT                 lr5 lr6 row_loop ;;

    BKPT ;;
