// ============================================================================
// argmax_match_mt.asm  --  multi-tile one-hot (temperature hardmax) over C cols
// ----------------------------------------------------------------------------
// Multi-tile version of argmax_match.asm for a matching-matrix row wider than
// 128 (e.g. C=512/513). Emits the one-hot assignment row
//     e_c = 2^( x_c - max_{c'} x_{c'} )           (caller pre-scales x by T)
// and the peak value max_c x_c (aaq0). The global max over C>128 columns is
// folded across T=ceil(C/128) tiles (AGG max re-includes the running AAQ).
//
// Numeric mode: WIDE-VECTOR FP32.  Address of col-tile t = in_addr + t*tile_stride.
//
// Register / memory contract:
//   CR0  = 0                  (zero base)
//   CR1  = 1                  (one; loop increment)
//   CR2  = in_addr            row of C scores (already x T), T col-tiles
//   CR3  = out_addr           one-hot row written here (T tiles)
//   CR4  = ones_addr          128xf32 all-1.0
//   CR8  = num_col_tiles      T = ceil(C/128)
//   CR9  = float32_bits(-1.0) = 0xBF800000
//   CR10 = tile_stride        bytes between col-tiles (= 512)
//   CR11 = 128                lane count for AGG reductions
//   CR15 = dtype sentinel (INT8)
//
// Outputs: aaq0 = peak (max score in scaled units); one-hot row at out_addr.
// ============================================================================

    SET                 lr11 cr11 ;;        // 128
    SET                 lr8 cr8 ;;          // T
    SET                 lr0 cr0 ;;          // 0
    LDR_CYCLIC_MULT_REG lr0 cr4 lr0 ;;      // R_CYCLIC = ones

    // ---- global max over T tiles -> aaq0 ---------------------------------
    SET                 lr4 cr2 ;;          // tile addr = in_addr
    LDR_MULT_REG        r0 lr4 cr0 ;;       // R0 = tile0
    MULT.EE             r0 lr0 0 lr0 ;;      // m  = tile0
    ACC.FIRST ;;                            // a  = tile0
    AGG.FIRST           max value lr11 cr0 aaq0 ;;  // aaq0 = max(tile0)
    ADD                 lr4 lr4 cr10 ;;
    SET                 lr7 cr1 ;;          // tile counter = 1
mmax_loop:
    LDR_MULT_REG        r0 lr4 cr0 ;;
    MULT.EE             r0 lr0 0 lr0 ;;
    ACC.FIRST ;;
    AGG                 max value lr11 cr0 aaq0 ;;  // fold: aaq0 = max(tile_t, aaq0)
    ADD                 lr4 lr4 cr10 ;;
    ADD                 lr7 lr7 cr1 ;;
    BLT                 lr7 lr8 mmax_loop ;;

    // ---- aaq1 = -max -----------------------------------------------------
    MULT.VE.AAQ         lr0 0 lr0 aaq0 ;;    // m = max (broadcast)
    ACC.FIRST ;;                            // a = max
    AGG.FIRST           max value_cr lr11 cr9 aaq1 ;;  // aaq1 = -max

    // ---- one-hot per tile: e = 2^(tile - max) ; store --------------------
    SET                 lr4 cr2 ;;          // in tile addr
    SET                 lr9 cr3 ;;          // out tile addr
    SET                 lr7 cr0 ;;          // tile counter = 0
oh_loop:
    LDR_MULT_REG        r0 lr4 cr0 ;;       // R0 = tile
    MULT.EE             r0 lr0 0 lr0 ;;      // m  = tile
    ACC.ADD_AAQ.FIRST   aaq1 ;;             // a  = tile - max
    ACTIVATE            lr11 exp2 ;;         // POST_AAQ = one-hot tile
    STR_POST_AAQ_REG    lr9 cr0 ;;          // store one-hot tile
    ADD                 lr4 lr4 cr10 ;;
    ADD                 lr9 lr9 cr10 ;;
    ADD                 lr7 lr7 cr1 ;;
    BLT                 lr7 lr8 oh_loop ;;

    BKPT ;;
