    @Positive
  void basic_test() {
    @Positive
    int[] b = new int[4];
    @Positive
    @LTLengthOf("b") int[] a = {0, 1, 2, 3};

    // :: error: (array.initializer)::error: (assignment)
    @Positive
    @LTLengthOf("b") int[] a1 = {0, 1, 2, 4};

    @Positive
    @LTEqLengthOf("b") int[] c = {-1, 4, 3, 1};

    // :: error: (array.initializer)::error: (assignment)
    @Positive
    @LTEqLengthOf("b") int[] c2 = {-1, 4, 5, 1};
    @Positive
  }
