// ============================================================================
// pixel_shuffle.asm  --  depth-to-space plane relocation (heatmap reshape core)
// ----------------------------------------------------------------------------
// Layer covered:
//   * SuperPoint detector head reshape: after the channel softmax, the 64
//     (= 8x8) "no-keypoint-removed" channels are rearranged (depth-to-space /
//     pixel-shuffle) into the full-resolution heatmap.
//
// WHAT THIS KERNEL DOES: a strided plane copy. It moves K contiguous 128-lane
//   (512-byte) source planes to K destination planes at a configurable output
//   stride:
//       dst_addr + t*dst_stride  <-  src_addr + t*src_stride   (t = 0..K-1)
//   This relocates channel planes (the coarse, plane-granular part of
//   depth-to-space) using only contiguous loads/stores.
//
// WHAT IT DOES NOT DO: the fine within-plane interleave of pixel-shuffle
//   (scattering channel c's pixels to (2x,2y)+offset positions). That is a
//   per-element scatter with non-contiguous addresses, which the contiguous-only
//   LDR/STR cannot express; the host performs the final interleave, or supplies
//   src planes already laid out so that a plane copy completes the reshape.
//   See README LIMITATIONS.
//
// Numeric mode: WIDE-VECTOR FP32 (values are copied verbatim; arithmetic is x1).
//
// Register / memory contract:
//   CR0  = 0                  (zero base)
//   CR1  = 1                  (one; loop increment)
//   CR2  = src_addr           base of source planes
//   CR3  = dst_addr           base of destination planes
//   CR4  = ones_addr          128xf32 all-1.0
//   CR6  = K                  number of planes to copy (>=1)
//   CR7  = src_stride         byte stride between source planes (e.g. 512)
//   CR8  = dst_stride         byte stride between destination planes
//   CR15 = dtype sentinel (INT8)
// ============================================================================

    SET                 lr0 cr0 ;;          // lr0 = 0
    SET                 lr6 cr6 ;;          // lr6 = K
    SET                 lr5 cr0 ;;          // lr5 = 0  (plane counter)
    SET                 lr2 cr2 ;;          // lr2 = source address (moving)
    SET                 lr3 cr3 ;;          // lr3 = dest address (moving)

    LDR_CYCLIC_MULT_REG lr0 cr4 lr0 ;;      // R_CYCLIC = ones

copy_loop:
    LDR_MULT_REG        r0 lr2 cr0 ;;       // R0       = src plane (addr = lr2)
    MULT.EE             r0 lr0 0 lr0 ;;      // MULT_RES = src * 1.0
    ACC.FIRST ;;                            // R_ACC    = src plane
    STR_ACC_REG         lr3 cr0 ;;          // dst plane = src plane (addr = lr3)

    ADD                 lr2 lr2 cr7 ;;      // next source plane
    ADD                 lr3 lr3 cr8 ;;      // next dest plane
    ADD                 lr5 lr5 cr1 ;;      // counter += 1
    BLT                 lr5 lr6 copy_loop ;;

    BKPT ;;
