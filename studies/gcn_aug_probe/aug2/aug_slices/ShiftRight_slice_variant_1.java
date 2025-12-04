/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

    @Positive
  void indexOrHigh(Object[] a, @IndexOrHigh("#1") int i) {
    @Positive
    @IndexOrHigh("a") int o = i >> 2;
    @Positive
    @IndexOrHigh("a") int p = i >>> 2;
    // Not true if a.length == 0
    // :: error: (assignment)
    @Positive
    @IndexFor("a") int q = i >> 2;
    @Positive
  }
