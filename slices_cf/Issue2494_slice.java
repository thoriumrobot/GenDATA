  static final long @MinLen(1) [] factorials = {
    1L,
    1L,
    1L * 2,
    1L * 2 * 3,
    1L * 2 * 3 * 4,
    1L * 2 * 3 * 4 * 5,
    1L * 2 * 3 * 4 * 5 * 6,
    1L * 2 * 3 * 4 * 5 * 6 * 7
  };

  static void binomialA(
    @Positive
      @NonNegative @LTLengthOf("Issue2494.factorials") int n,
      @NonNegative @LessThan("#1 + 1") int k) {
    @NonNegative
    @IndexFor("factorials") int j = k;
  }
}
