  public static long binomial(
    @Positive
      @NonNegative @LTLengthOf("BinomialTest.factorials") int n,
      @NonNegative @LessThan("#1 + 1") int k) {
    return factorials[k];
  }

  public static void binomial0(
    @Positive
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1") int k) {
    @Positive
    @LTLengthOf(value = "factorials", offset = "1") int i = k;
  }

  public static void binomial0Error(
    @Positive
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1") int k) {
    // :: error: (assignment)
    @Positive
    @LTLengthOf(value = "factorials", offset = "2") int i = k;
  }

  public static void binomial0Weak(
    @Positive
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1") int k) {
    @Positive
    @LTLengthOf("factorials") int i = k;
  }

  public static void binomial1(
    @Positive
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1 + 1") int k) {
    @Positive
    @LTLengthOf("factorials") int i = k;
  }

  public static void binomial1Error(
    @Positive
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1 + 1") int k) {
    // :: error: (assignment)
    @Positive
    @LTLengthOf(value = "factorials", offset = "1") int i = k;
  }

  public static void binomial2(
    @Positive
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1 + 2") int k) {
    @Positive
    @LTLengthOf(value = "factorials", offset = "-1") int i = k;
  }

  public static void binomial2Error(
    @Positive
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1 + 2") int k) {
    // :: error: (assignment)
    @Positive
    @LTLengthOf(value = "factorials", offset = "0") int i = k;
  }

  public static void binomial_1(
    @Positive
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1 - 1") int k) {
    @Positive
    @LTLengthOf(value = "factorials", offset = "2") int i = k;
  }
