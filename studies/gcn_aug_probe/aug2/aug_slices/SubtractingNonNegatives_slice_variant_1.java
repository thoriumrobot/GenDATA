/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

    @Positive
  public static void m4(int[] a, @IndexFor("#1") int i, @IndexFor("#1") int j) {
    @Positive
    int k = i;
    @Positive
    if (k >= j) {
    @Positive
      @IndexFor("a") int y = k;
    @Positive
    }
    @Positive
    for (k = i; k >= j; k -= j) {
    @Positive
      @IndexFor("a") int x = k;
    @Positive
    }
    @Positive
  }
