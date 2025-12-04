/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

    @Positive
  public static void m2(int @MinLen(1) [] shorter) {
    @Positive
    int[] longer = new int[shorter.length * 2];
    @Positive
    for (int i = 0; i < shorter.length; i++) {
    @Positive
      longer[i] = shorter[i];
    @Positive
    }
    @Positive
  }

    @Positive
  public static void m3(int[] shorter) {
    @Positive
    int[] longer = new int[shorter.length + 1];
    @Positive
    for (int i = 0; i < shorter.length; i++) {
    @Positive
      longer[i] = shorter[i];
    @Positive
    }
    @Positive
  }
