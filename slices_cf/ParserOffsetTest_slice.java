  @SuppressWarnings("lowerbound")
    @NonNegative
  public void subtraction4(String[] a, @IndexFor("#1") int i) {
    if (1 - i < a.length) {
      // The error on this assignment is a false positive.
      // :: error: (assignment)
    @NonNegative
      @IndexFor("a") int j = 1 - i;

      // :: error: (assignment)
    @Positive
      @LTLengthOf(value = "a", offset = "1") int k = i;
    }
  }

  @SuppressWarnings("lowerbound")
  public void subtraction5(String[] a, int i) {
    if (1 - i < a.length) {
      // :: error: (assignment)
    @NonNegative
      @IndexFor("a") int j = i;
    }
  }

  @SuppressWarnings("lowerbound")
  public void subtraction6(String[] a, int i, int j) {
    if (i - j < a.length - 1) {
    @NonNegative
      @IndexFor("a") int k = i - j;
      // :: error: (assignment)
    @NonNegative
      @IndexFor("a") int k1 = i;
    }
  }

  public void multiplication1(String[] a, int i, @Positive int j) {
    if ((i * j) < (a.length + j)) {
      // :: error: (assignment)
    @NonNegative
      @IndexFor("a") int k = i;
      // :: error: (assignment)
    @NonNegative
      @IndexFor("a") int k1 = j;
    }
  }

  public void multiplication2(String @ArrayLen(5) [] a, @IntVal(-2) int i, @IntVal(20) int j) {
    if ((i * j) < (a.length - 20)) {
    @Positive
      @LTLengthOf("a") int k1 = i;
      // :: error: (assignment)
    @Positive
      @LTLengthOf(value = "a", offset = "20") int k2 = i;
      // :: error: (assignment)
    @Positive
      @LTLengthOf("a") int k3 = j;
    }
  }
