    @Positive
  void foo(int[] a, @LengthOf("#1") int x) {
    @Positive
    @IndexOrHigh("a") int y = x;
    // :: error: (assignment)
    @Positive
    @IndexFor("a") int w = x;
    @Positive
    @LengthOf("a") int z = a.length;
    @Positive
  }
