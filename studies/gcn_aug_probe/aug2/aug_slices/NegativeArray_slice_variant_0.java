/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_logical_expression

    @Positive
  public static void negativeArray(@GTENegativeOne int len) {
    // :: error: (array.length.negative)
    @Positive
    int[] arr = new int[len];
    @Positive
  }
