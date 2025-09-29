    @Positive
  void testLessThanLength(String[] s, @IndexOrHigh("#1") int i, @IndexOrHigh("#1") int j) {
    if (i < Array.getLength(s)) {
    @NonNegative
      @IndexFor("s") int in = i;
      // ::  error: (assignment)
    @NonNegative
      @IndexFor("s") int jn = j;
    }
  }
