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

    int b = 2;
    if (test >= b) {
    @Positive
      @LTLengthOf("arr") int c = b;
    }
    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int c1 = b;

    if (a >= b) {
      int potato = 7;
    } else {
      // :: error: (assignment)
    @Positive
      @LTLengthOf("arr") int d = b;
    }
  }

    @Positive
  void testLTEL(@LTEqLengthOf("arr") int test) {
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int a = Integer.parseInt("1");

    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int a3 = Integer.parseInt("3");

    int b = 2;
    if (test >= b) {
    @Positive
      @LTEqLengthOf("arr") int c = b;
    }
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int c1 = b;

    if (a >= b) {
      int potato = 7;
    } else {
      // :: error: (assignment)
    @Positive
      @LTEqLengthOf("arr") int d = b;
    }
  }
