/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_logical_expression

    @Positive
  void testLTL(@LTLengthOf("arr") int test) {
    // The reason for the parsing is so that the Value Checker
    // can't figure it out but normal humans can.

    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int a = Integer.parseInt("1");

    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int a3 = Integer.parseInt("3");

    @Positive
    int b = 2;
    @Positive
    if (b <= test) {
    @Positive
      @LTLengthOf("arr") int c = b;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int c1 = b;

    @Positive
    if (b <= a) {
    @Positive
      int potato = 7;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @LTLengthOf("arr") int d = b;
    @Positive
    }
    @Positive
  }

    @Positive
  void testLTEL(@LTEqLengthOf("arr") int test) {
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int a = Integer.parseInt("1");

    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int a3 = Integer.parseInt("3");

    @Positive
    int b = 2;
    @Positive
    if (b <= test) {
    @Positive
      @LTEqLengthOf("arr") int c = b;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int c1 = b;

    @Positive
    if (b <= a) {
    @Positive
      int potato = 7;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @LTLengthOf("arr") int d = b;
    @Positive
    }
    @Positive
  }
