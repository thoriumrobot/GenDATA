    @NonNegative
  public static void bug2(int[] a, @IndexFor("#1") int i, @IndexFor("#1") int j) {
    @Positive
    @LTLengthOf("a") int k = ((i - 1) + j) / 2;
    // :: error: (assignment)
    @Positive
    @LTLengthOf("a") int h = ((i + 1) + j) / 2;
  }
