/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

    @Positive
  void m1(Object[] a, @IndexOrHigh("#1") int i, @NonNegative int j) {
    @Positive
    @IndexFor("a") int k = j % i;
    @Positive
  }
