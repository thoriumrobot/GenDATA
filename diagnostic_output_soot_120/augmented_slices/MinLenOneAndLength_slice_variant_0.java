/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_logical_expression

    @Positive
  public void m1(int @MinLen(1) [] a, int[] b) {
    @Positive
    @IndexFor("a") int i = a.length / 2;
    // :: error: (assignment)
    @Positive
    @IndexFor("b") int j = b.length / 2;
    @Positive
  }
