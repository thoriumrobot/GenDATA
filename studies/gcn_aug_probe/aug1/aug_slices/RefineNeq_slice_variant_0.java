/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_logical_expression

    @Positive
  void testLTL(@LTLengthOf("arr") int test) {
    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int a = Integer.parseInt("1");

    @Positive
    int b = 1;
    @Positive
    if (test != b) {
      // :: error: (assignment)
    @Positive
      @LTLengthOf("arr") int e = b;

    @Positive
    } else {

    @Positive
      @LTLengthOf("arr") int c = b;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int d = b;
    @Positive
  }

    @Positive
  void testLTEL(@LTEqLengthOf("arr") int test) {
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int a = Integer.parseInt("1");

    @Positive
    int b = 1;
    @Positive
    if (test != b) {
      // :: error: (assignment)
    @Positive
      @LTEqLengthOf("arr") int e = b;
    @Positive
    } else {
    @Positive
      @LTEqLengthOf("arr") int c = b;

    @Positive
      @LTLengthOf("arr") int g = b;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int d = b;
    @Positive
  }
