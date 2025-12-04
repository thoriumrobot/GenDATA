/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

    @Positive
  void test(int[] a, boolean cond) {
    @Positive
    int[] b;
    @Positive
    if (cond) {
    @Positive
      b = a;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    int @SameLen({"a", "b"}) [] c = a;
    @Positive
  }
