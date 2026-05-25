// ============================================================================
// maxpool_shift.asm  --  3x3 sliding-window max-pool, GATHER-FREE, host-free
// ----------------------------------------------------------------------------
// Fixed-window max-pool / NMS-pool done with shifted CONTIGUOUS loads instead
// of a host gather. For a fixed image size the window taps are constant
// flat-index offsets, so tap_t for output tile p is just a contiguous load at
//     addr = heatmap_base + p + (dy*rowbytes + dx*elem)
// The kernel sweeps all output tiles (outer loop) and the 3x3 window
// (dy,dx loops) on-device -- one launch, start to finish, no host gather and no
// per-tap plane materialization.
//
//   y[p_lane] = max over dy,dx in {0,1,2} of  heatmap[p_lane + dy*W + dx]
//
// Layout (precomputed offline for the fixed size): the H x W heatmap is stored
// in a buffer padded by a 1-px border (filled -inf). Output row y is one 128-
// lane tile whose top-left window corner is padded row y; the window center
// (dy=1,dx=1) is the original pixel. Border padding = -inf so out-of-image taps
// never win the max. Per frame the host just writes pixels into the padded
// buffer interior (normal staging) -- no reshuffle.
//
// Numeric mode: WIDE-VECTOR FP32.
//
// Register / memory contract:
//   CR0=0, CR1=1
//   CR2  = heatmap_base   padded heatmap (Hp rows x 128, border = -inf)
//   CR3  = out_base       pooled output (num_tiles rows x 128)
//   CR4  = ones_addr      128xf32 all-1.0
//   CR5  = neginf_addr    128xf32 all -3.4e38   (R_ACC seed)
//   CR6  = num_tiles      number of output tiles (= output rows)
//   CR7  = tile_stride    bytes between output tiles (= padded row = 512)
//   CR8  = rowbytes       bytes per padded heatmap row (= 512)
//   CR9  = window_dim     3  (loop bound for dy/dx)
//   CR10 = 128            valid lane count for the seed reduction
//   CR11 = elem_bytes     4
//   CR15 = dtype sentinel (INT8)
//
// Output: out_base[tile] = 3x3 max-pool of the heatmap (one launch, host-free).
// ============================================================================

    SET                 lr0 cr0 ;;          // 0
    SET                 lr10 cr6 ;;          // num_tiles
    SET                 lr11 cr9 ;;          // window_dim = 3
    LDR_CYCLIC_MULT_REG lr0 cr4 lr0 ;;       // R_CYCLIC = ones

    // ---- aaq0 = -inf (neutral for 3-way ACC.MAX) -------------------------
    LDR_MULT_REG        r0 lr0 cr5 ;;
    MULT.EE             r0 lr0 0 lr0 ;;
    ACC.FIRST ;;
    AGG.FIRST           max value cr10 cr0 aaq0 ;;   // aaq0 = -inf

    SET                 lr2 cr0 ;;           // p = output/tile byte offset = 0
    SET                 lr3 cr0 ;;           // tile counter = 0

tile_loop:
    // ---- seed R_ACC = -inf -------------------------------------------------
    LDR_MULT_REG        r0 lr0 cr5 ;;        // R0 = -inf plane
    MULT.EE             r0 lr0 0 lr0 ;;
    ACC.FIRST ;;                             // R_ACC = -inf

    // ---- 3x3 window: dy in 0..2, dx in 0..2 ------------------------------
    SET                 lr4 cr0 ;;           // dy = 0
    SET                 lr5 cr0 ;;           // rowsh = 0
dy_loop:
    SET                 lr6 cr0 ;;           // dx = 0
    SET                 lr7 cr0 ;;           // colsh = 0
dx_loop:
    ADD                 lr8 lr2 lr5 ;;       // off = p + rowsh
    ADD                 lr8 lr8 lr7 ;;       // off += colsh
    LDR_MULT_REG        r0 lr8 cr2 ;;        // R0 = heatmap[base + off]
    MULT.EE             r0 lr0 0 lr0 ;;      // m = tap
    ACC.MAX             aaq0 ;;              // R_ACC = max(R_ACC, tap, -inf)
    ADD                 lr7 lr7 cr11 ;;      // colsh += elem (4)
    ADD                 lr6 lr6 cr1 ;;       // dx++
    BLT                 lr6 lr11 dx_loop ;;
    ADD                 lr5 lr5 cr8 ;;       // rowsh += rowbytes
    ADD                 lr4 lr4 cr1 ;;       // dy++
    BLT                 lr4 lr11 dy_loop ;;

    STR_ACC_REG         lr2 cr3 ;;           // out[base_out + p] = pooled max
    ADD                 lr2 lr2 cr7 ;;       // next output tile
    ADD                 lr3 lr3 cr1 ;;       // tile counter++
    BLT                 lr3 lr10 tile_loop ;;

    BKPT ;;
