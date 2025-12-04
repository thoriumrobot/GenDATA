/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_logical_expression

    @Positive
  public static long binomial(
    @Positive
      @NonNegative @LTLengthOf("BinomialTest.factorials") int n,
    @Positive
      @NonNegative @LessThan("#1 + 1") int k) {
    @Positive
    return factorials[k];
    @Positive
  }

    @Positive
  public static void binomial0(
    @Positive
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1") int k) {
    @Positive
    @LTLengthOf(value = "factorials", offset = "1") int i = k;
    @Positive
  }

    @Positive
  public static void binomial0Error(
    @Positive
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1") int k) {
    // :: error: (assignment)
    @Positive
    @LTLengthOf(value = "factorials", offset = "2") int i = k;
    @Positive
  }

    @Positive
  public static void binomial0Weak(
    @Positive
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1") int k) {
    @Positive
    @LTLengthOf("factorials") int i = k;
    @Positive
  }

    @Positive
  public static void binomial1(
    @Positive
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1 + 1") int k) {
    @Positive
    @LTLengthOf("factorials") int i = k;
    @Positive
  }
