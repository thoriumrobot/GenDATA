/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

    @Positive
  void subtype(int @SameLen("#2") [] a, int[] b) {
    @Positive
    int @SameLen({"a", "b"}) [] c = a;

    // :: error: (assignment)
    @Positive
    int @SameLen("c") [] q = {1, 2};
    @Positive
    int @SameLen("c") [] d = q;

    // :: error: (assignment)
    @Positive
    int @SameLen("f") [] e = a;
    @Positive
  }
