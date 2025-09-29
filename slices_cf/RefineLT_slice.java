    @Positive
  void testLTL(@LTLengthOf("arr") int test, @LTLengthOf("arr") int a, @LTLengthOf("arr") int a3) {
    int b = 2;
    if (b < test) {
    @Positive
      @LTLengthOf("arr") int c = b;
    }
    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int c1 = b;

    if (b < a3) {
      int potato = 7;
    } else {
      // :: error: (assignment)
    @Positive
      @LTLengthOf("arr") int d = b;
    }
  }

    @Positive
  void testLTEL(@LTLengthOf("arr") int test) {
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int a = Integer.parseInt("1");

    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int a3 = Integer.parseInt("3");

    int b = 2;
    if (b < test) {
    @Positive
      @LTEqLengthOf("arr") int c = b;
    }
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int c1 = b;

    if (b < a) {
      int potato = 7;
    } else {
      // :: error: (assignment)
    @Positive
      @LTEqLengthOf("arr") int d = b;
    }
  }
