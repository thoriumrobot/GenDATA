/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

    @Positive
  public static void m1(int[] shorter) {
    @Positive
    int[] longer = new int[4 * shorter.length];
    // :: error: (assignment)
    @Positive
    @LTLengthOf("longer") int x = shorter.length;
    @Positive
    int i = longer[x];
    @Positive
  }
