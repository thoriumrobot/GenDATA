  void test2(int[] array) {
    @Positive
    int len = array.length;
    @Positive
    int lenM1 = array.length - 1;
    @Positive
    int lenP1 = array.length + 1;
    // :: error: (assignment)
    @Positive
    @LTLengthOf("array") int x = len / 2;
    @Positive
    @LTLengthOf("array") int y = lenM1 / 3;
    @Positive
    @LTEqLengthOf("array") int z = len / 1;
    // :: error: (assignment)
    @Positive
    @LTLengthOf("array") int w = lenP1 / 2;
  }
