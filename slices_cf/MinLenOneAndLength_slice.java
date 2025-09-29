  public void m1(int @MinLen(1) [] a, int[] b) {
    @Positive
    @IndexFor("a") int i = a.length / 2;
    // :: error: (assignment)
    @Positive
    @IndexFor("b") int j = b.length / 2;
  }
