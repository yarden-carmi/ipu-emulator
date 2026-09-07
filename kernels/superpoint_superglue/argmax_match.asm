// ============================================================================
// argmax_match.asm  --  hard-argmax (one-hot) over a score row, for matching
// ----------------------------------------------------------------------------
// Layer covered:
//   * SuperGlue matching read-out: turning a soft assignment row into a hard
//     argmax / mutual-nearest-neighbour decision. Running this over rows and
//     over columns (P^T) and AND-ing the two one-hots gives mutual matches.
//
// WHY ONE-HOT INSTEAD OF AN INTEGER INDEX: the ISA cannot move a vector lane
//   into a scalar register, and branches read only LR/CR scalars, so the
//   integer argmax index is not extractable on-device (see README LIMITATIONS).
//   Instead we emit a one-hot VECTOR e where e[argmax] ~ 1 and e[else] ~ 0,
//   which is exactly the assignment-matrix row SuperGlue needs.
//
// Method (temperature hardmax):  e_i = 2^( T*(x_i - max_j x_j) ).
//   At the argmax the exponent is 0 -> e = 1; elsewhere it is <= -T -> e ~ 0.
//   T is the "temperature". The CALLER pre-scales the row by T (stores T*x), so
//   this kernel sees scaled logits and computes 2^((T x) - max(T x)). A larger
//   T sharpens toward a true one-hot (ties split mass equally).
//
// Numeric mode: WIDE-VECTOR FP32.
//
// Register / memory contract:
//   CR0  = 0                       (zero base)
//   CR2  = row_addr                Nxf32 input row, ALREADY scaled by T
//   CR3  = out_addr                Nxf32 one-hot assignment row
//   CR4  = ones_addr               128xf32 all-1.0
//   CR6  = N                       valid lanes (1..128)
//   CR7  = float32_bits(-1.0)      = 0xBF800000
//   CR15 = dtype sentinel (INT8)
// ============================================================================

    SET                 lr6 cr6 ;;          // lr6 = N
    SET                 lr0 cr0 ;;          // lr0 = 0

    // ---- scaled row (T*x) into R_ACC -------------------------------------
    LDR_MULT_REG        r0 lr0 cr2 ;;       // R0       = T*x
    LDR_CYCLIC_MULT_REG lr0 cr4 lr0 ;;      // R_CYCLIC = ones
    MULT.EE             r0 lr0 0 lr0 ;;      // MULT_RES = T*x
    ACC.FIRST ;;                            // R_ACC    = T*x

    // ---- aaq0 = -max(T*x) ------------------------------------------------
    AGG.FIRST           max value_cr lr6 cr7 aaq0 ;;   // max * (-1)

    // ---- R_ACC = T*x - max(T*x)  (MULT_RES still = T*x) ------------------
    ACC.ADD_AAQ.FIRST   aaq0 ;;

    // ---- one-hot = 2^(T*(x - argmax)) ------------------------------------
    ACTIVATE            lr6 exp2 ;;          // POST_AAQ = one-hot row
    STR_POST_AAQ_REG    lr0 cr3 ;;
    BKPT ;;
