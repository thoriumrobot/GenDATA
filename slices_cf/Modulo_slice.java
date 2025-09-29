    @NonNegative
  void m1(Object[] a, @IndexOrHigh("#1") int i, @NonNegative int j) {
    @NonNegative
    @IndexFor("a") int k = j % i;
  }

    @Positive
  void m1p(Object[] a, @Positive @LTEqLengthOf("#1") int i, @Positive int j) {
    @NonNegative
    @IndexFor("a") int k = j % i;
  }
