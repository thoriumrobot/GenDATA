  void m1(Object[] a, @IndexOrHigh("#1") int i, @NonNegative int j) {
    @IndexFor("a") int k = j % i;
  }
