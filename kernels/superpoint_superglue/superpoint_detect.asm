// ============================================================================
// superpoint_detect.asm  --  detector confidence + fixed-threshold selection
//                            (cells-in-lanes, VLIW-packed), host-free
// ----------------------------------------------------------------------------
// Per output tile = 128 cells in lanes; the Nc detector channels are separate
// planes (channel-major, the natural conv output layout). Computes per cell:
//     conf = max_c logit_c                         (detection score)
//     keep = relu(conf - tau)                      (fixed-threshold selection)
// Output is DENSE (one f32 per cell), no broadcast-store. The per-channel max
// is one PACKED compound { LR addr ; LR ch++ ; LDR_CYCLIC chan ; MULT.EE(ones)
// ; ACC.MAX ; BLT } -> ACC.MAX busy every channel cycle.
//
// (Sub-pixel argmax coords use the per-cell cell_nms kernel on survivors; this
// kernel delivers the dense detection/selection map that feeds top-k.)
//
// Layout: channel c of tile t at IN + c*chan_stride + t*512 (chan_stride =
// num_tiles*512 = all cells of one channel). conf/keep dense at OUT + t*512.
//
// Register / memory contract:
//   CR0=0, CR1=1
//   CR2  = IN base (channel-major detector logits)
//   CR3  = OUT base (dense per-cell keep = relu(conf-tau))
//   CR4  = chan_stride (bytes per channel plane)
//   CR5  = tile_stride (= 512)
//   CR6  = num_tiles
//   CR7  = Nc (number of channels)
//   CR8  = ones_addr (128xf32 = 1.0)
//   CR9  = neginf_addr (128xf32 = -3.4e38)
//   CR10 = negtau_addr (128xf32 = -tau)
//   CR11 = 128 (valid lanes for relu)
//   CR15 = dtype sentinel (INT8)
// ============================================================================

    SET                 lr11 cr11 ;;        // 128
    SET                 lr0 cr0 ;;           // 0
    LDR_MULT_REG        r0 lr0 cr8 ;;        // R0 = ones (prologue; stable)

    // seed aaq0 = -inf
    LDR_CYCLIC_MULT_REG lr0 cr9 lr0 ;;       // R_CYCLIC = -inf
    MULT.EE             r0 lr0 0 lr0 ;;
    ACC.FIRST ;;
    AGG.FIRST           max value lr11 cr0 aaq0 ;;   // aaq0 = -inf

    SET                 lr1 cr0 ;;           // lr1 = tile byte offset = 0
    SET                 lr2 cr0 ;;           // lr2 = tile counter = 0

tile_loop:
    // ---- channel 0 -> R_ACC ----------------------------------------------
    ADD                 lr3 lr1 cr0 ;;       // lr3 = chan addr = IN_tile (=tile offset)
    LDR_CYCLIC_MULT_REG lr3 cr2 lr0 ;;       // R_CYCLIC = chan0 (addr = lr3 + IN)
    MULT.EE             r0 lr0 0 lr0 ;;       // mult_res = ones*chan0 = chan0
    ACC.FIRST ;;                             // R_ACC = chan0
    SET                 lr4 cr1 ;;           // ch counter = 1

    // ---- max over channels 1..Nc-1 (one packed compound per channel) -----
ch_loop:
    ADD lr3 lr3 cr4 ; ADD lr4 lr4 cr1 ; LDR_CYCLIC_MULT_REG lr3 cr2 lr0 ; MULT.EE r0 lr0 0 lr0 ; ACC.MAX aaq0 ; BLT lr4 cr7 ch_loop ;;

    // ---- keep = relu(conf - tau) -----------------------------------------
    LDR_CYCLIC_MULT_REG lr0 cr10 lr0 ;;      // R_CYCLIC = -tau plane
    MULT.EE             r0 lr0 0 lr0 ;;       // mult_res = -tau
    ACC ;;                                   // R_ACC = conf + (-tau) = conf - tau
    ACTIVATE            lr11 relu ;;          // POST_AAQ = relu(conf - tau)
    STR_POST_AAQ_REG    lr1 cr3 ;;            // OUT[tile] = keep (dense)

    ADD                 lr1 lr1 cr5 ;;        // next tile
    ADD                 lr2 lr2 cr1 ;;
    BLT                 lr2 cr6 tile_loop ;;

    BKPT ;;
