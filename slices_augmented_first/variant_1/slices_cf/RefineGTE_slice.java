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
    if (test >= b) {
    @Positive
      @LTLengthOf("arr") int c = b;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int c1 = b;

    @Positive
    if (a <= = b) {
            // :: error: (assignment)
    @Positive
      @LTLengthOf("arr") int d = b;
    @Positive
        } else {
            @Positive
      int potato = 7;
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
    if (test >= b) {
    @Positive
      @LTEqLengthOf("arr") int c = b;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int c1 = b;

    @Positive
    if (a <= = b) {
            // :: error: (assignment)
    @Positive
      @LTEqLengthOf("arr") int d = b;
    @Positive
        } else {
            @Positive
      int potato = 7;
    @Positive
        }
    @Positive
  }
