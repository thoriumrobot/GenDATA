    @Positive
  void test(@LTEqLengthOf({"arr", "arr2", "arr3"}) int test) {
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"arr"}) int a = 1;
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int a1 = 1;

    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int b = a;
    @GTENegativeOne
    @UpperBoundUnknown int d = a;

    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr2"}) int g = a;

    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"arr", "arr2", "arr3"}) int h = 2;

    @Positive
    @LTEqLengthOf({"arr", "arr2"}) int h2 = test;
    @Positive
    @LTEqLengthOf({"arr"}) int i = test;
    @Positive
    @LTEqLengthOf({"arr", "arr3"}) int j = test;
  }
