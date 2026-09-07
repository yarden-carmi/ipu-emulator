// ============================================================================
// attention_scores.asm  --  one scaled dot-product score  s = (q . k)/sqrt(d)
// ----------------------------------------------------------------------------
// Layer covered:
//   * SuperGlue attentional GNN: the QK^T score computation inside self- and
//     cross-attention (feeds softmax.asm).
//
// Scope of ONE invocation: a single query q against a single key k, each with
//   d <= 128 dims. The scalar score is broadcast across the output plane so the
//   caller can read it from out_addr[0]. To score a query against M keys the
//   host advances key_addr / out_addr and re-invokes (the ISA has no
//   lane-indexed scatter to pack M scalars into M lanes of one vector -- see
//   README LIMITATIONS).
//
// Numeric mode: WIDE-VECTOR FP32.
//
// Method:
//   MULT.EE q_i * k_i -> ACC.FIRST            (R_ACC = elementwise products)
//   AGG sum value_cr  -> aaq0 = (q.k) * (1/sqrt(d))   (scale via CR float bits)
//   MULT.VE.AAQ over ones -> broadcast score -> store plane.
//
// Register / memory contract:
//   CR0  = 0                          (zero base)
//   CR2  = query_addr                 dxf32 query vector q
//   CR3  = key_addr                   dxf32 key vector k
//   CR4  = ones_addr                  128xf32 all-1.0
//   CR5  = out_addr                   score broadcast plane (read out[0])
//   CR7  = float32_bits(1/sqrt(d))    scale applied via value_cr
//   CR8  = d                          dimension / valid lanes (1..128)
//   CR15 = dtype sentinel (INT8)
// ============================================================================

    SET                 lr8 cr8 ;;          // lr8 = d
    SET                 lr0 cr0 ;;          // lr0 = 0

    LDR_MULT_REG        r0 lr0 cr2 ;;       // R0       = q
    LDR_CYCLIC_MULT_REG lr0 cr3 lr0 ;;      // R_CYCLIC = k
    MULT.EE             r0 lr0 0 lr0 ;;      // MULT_RES = q_i * k_i
    ACC.FIRST ;;                            // R_ACC    = products

    AGG.FIRST           sum value_cr lr8 cr7 aaq0 ;;  // aaq0 = (q.k)/sqrt(d)

    LDR_CYCLIC_MULT_REG lr0 cr4 lr0 ;;      // R_CYCLIC = ones
    MULT.VE.AAQ         lr0 0 lr0 aaq0 ;;    // MULT_RES = score (broadcast)
    ACC.FIRST ;;                            // R_ACC    = score in all lanes

    STR_ACC_REG         lr0 cr5 ;;          // out[0..] = score
    BKPT ;;
