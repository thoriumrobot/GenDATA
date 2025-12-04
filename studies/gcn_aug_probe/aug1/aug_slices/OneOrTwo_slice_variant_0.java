/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_logical_expression

    @Positive
  void test(@BottomVal int x) {
    @Positive
    int[] a = new int[Integer.valueOf(getOneOrTwo())];
    // :: error: (array.length.negative)
    @Positive
    int[] b = new int[Integer.valueOf(x)];
    @Positive
  }
