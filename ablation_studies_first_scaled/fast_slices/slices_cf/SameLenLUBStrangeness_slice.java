    @Positive
  void test(int[] a, boolean cond) {
    @Positive
    int[] b;
    @Positive
    if (cond) {
    @Positive
      b = a;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    int @SameLen({"a", "b"}) [] c = a;
    @Positive
  }
