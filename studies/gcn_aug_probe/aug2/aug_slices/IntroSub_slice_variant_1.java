/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

    @Positive
  void test(int[] arr) {
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int a = 3;
    @Positive
  }

    @Positive
  void test(int[] arr, @LTLengthOf({"#1"}) int a) {
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int c = a - (-1);
    @Positive
    @LTEqLengthOf({"arr"}) int c1 = a - (-1);
    @Positive
    @LTLengthOf({"arr"}) int d = a - 0;
    @Positive
    @LTLengthOf({"arr"}) int e = a - 7;
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int f = a - (-7);

    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"arr"}) int j = 7;
    @Positive
  }
