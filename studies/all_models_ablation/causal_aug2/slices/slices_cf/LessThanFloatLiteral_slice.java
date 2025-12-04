    @Positive
  void test(int x) {
    @Positive
    if (1.0 > x) {
      // TODO: It might be nice to handle comparisons against floats,
      // but an array index is not generally compared to a float.
      // :: error: (assignment)
    @Positive
      @LessThan("1") int y = x;
    @Positive
    }
    @Positive
  }
