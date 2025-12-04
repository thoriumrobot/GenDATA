/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_logical_expression

    @Positive
  void callTest1(int x) {
    @Positive
    test1(0);
    @Positive
    test1(1);
    @Positive
    test1(2);
    @Positive
    test1(array.length);
    // :: error: (argument)
    @Positive
    test1(array.length - 1);
    @Positive
    if (array.length > x) {
      // :: error: (argument)
    @Positive
      test1(x);
    @Positive
    }

    @Positive
    if (array.length == x) {
    @Positive
      test1(x);
    @Positive
    }
    @Positive
  }
