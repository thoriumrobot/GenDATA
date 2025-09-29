    @NonNegative
  public static void m2(int[] a, @IndexFor("#1") int i, @IndexFor("#1") int j) {
    // :: error: (assignment)
    @Positive
    @LTLengthOf("a") int h = ((i + 1) + j) >> 1;
  }
