/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_logical_expression

    @Positive
  void foo(int[] a, @LengthOf("#1") int x) {
    @Positive
    @IndexOrHigh("a") int y = x;
    // :: error: (assignment)
    @Positive
    @IndexFor("a") int w = x;
    @Positive
    @LengthOf("a") int z = a.length;
    @Positive
  }
