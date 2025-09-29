    @NonNegative
  void indexOrHigh(Object[] a, @IndexOrHigh("#1") int i) {
    @NonNegative
    @IndexOrHigh("a") int o = i >> 2;
    @NonNegative
    @IndexOrHigh("a") int p = i >>> 2;
    // Not true if a.length == 0
    // :: error: (assignment)
    @NonNegative
    @IndexFor("a") int q = i >> 2;
  }

    @Positive
  void negative(Object[] a, @LTLengthOf(value = "#1", offset = "100") int i) {
    // Not true for some negative i
    // :: error: (assignment)
    @Positive
    @LTLengthOf(value = "#1", offset = "100") int q = i >> 2;
  }
