/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_logical_expression

    @Positive
  void gte_check(int[] a) {
    @Positive
    if (a.length >= 1) {
    @Positive
      int @MinLen(1) [] b = a;
    @Positive
    }
    @Positive
  }
