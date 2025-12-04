/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

    @Positive
  void int_Integer_access_equivalent(@IndexFor("#3") Integer i, @IndexFor("#3") int j, int[] a) {
    @Positive
    a[i] = a[j];
    @Positive
  }
