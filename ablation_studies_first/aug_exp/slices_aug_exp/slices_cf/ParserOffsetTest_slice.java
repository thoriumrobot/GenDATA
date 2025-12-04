    @Positive
  public void addition3(String[] a, @IndexFor("#1") int i) {
    @Positive
    if ((i + 5) < a.length) {
    @Positive
      @IndexFor("a") int j = i + 5;
    @Positive
    }
    @Positive
  }

    @Positive
  public void subtraction3(String[] a, @NonNegative int k) {
    @Positive
    if (k - 5 < a.length) {
    @Positive
      String s = a[k - 5];
    @Positive
      @IndexFor("a") int j = k - 5;
    @Positive
    }
    @Positive
  }

    @Positive
  public void subtraction4(String[] a, @IndexFor("#1") int i) {
    @Positive
    if (1 - i < a.length) {
      // The error on this assignment is a false positive.
      // :: error: (assignment)
    @Positive
      @IndexFor("a") int j = 1 - i;

      // :: error: (assignment)
    @Positive
      @LTLengthOf(value = "a", offset = "1") int k = i;
    @Positive
    }
    @Positive
  }

    @Positive
  public void subtraction5(String[] a, int i) {
    @Positive
    if (1 - i < a.length) {
      // :: error: (assignment)
    @Positive
      @IndexFor("a") int j = i;
    @Positive
    }
    @Positive
  }
