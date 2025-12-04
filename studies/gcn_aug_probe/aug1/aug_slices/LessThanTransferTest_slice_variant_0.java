/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_logical_expression

    @Positive
  void lt_check(int[] a) {
    @Positive
    if (0 < a.length) {
    @Positive
      int @MinLen(1) [] b = a;
    @Positive
    }
    @Positive
  }
