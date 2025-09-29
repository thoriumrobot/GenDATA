    @Positive
  void foo(int[] a, @LengthOf("#1") int x) {
    @NonNegative
    @IndexOrHigh("a") int y = x;
    // :: error: (assignment)
    @NonNegative
    @IndexFor("a") int w = x;
    @Positive
    @LengthOf("a") int z = a.length;
  }
