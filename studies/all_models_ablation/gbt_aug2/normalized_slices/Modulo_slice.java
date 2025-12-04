    @Positive
  void m1(Object[] a, @IndexOrHigh("#1") int i, @NonNegative int j) {
    @Positive
    @IndexFor("a") int k = j % i;
    @Positive
  }
