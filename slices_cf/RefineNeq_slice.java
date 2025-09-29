    @Positive
  void testLTL(@LTLengthOf("arr") int test) {
    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int a = Integer.parseInt("1");

    int b = 1;
    if (test != b) {
      // :: error: (assignment)
    @Positive
      @LTLengthOf("arr") int e = b;

    } else {

    @Positive
      @LTLengthOf("arr") int c = b;
    }
    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int d = b;
  }

    @Positive
  void testLTEL(@LTEqLengthOf("arr") int test) {
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int a = Integer.parseInt("1");

    int b = 1;
    if (test != b) {
      // :: error: (assignment)
    @Positive
      @LTEqLengthOf("arr") int e = b;
    } else {
    @Positive
      @LTEqLengthOf("arr") int c = b;

    @Positive
      @LTLengthOf("arr") int g = b;
    }
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int d = b;
  }
