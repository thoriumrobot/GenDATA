/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

    @Positive
  public static void OffsetStringBuilder() {
    @Positive
    StringBuilder stringBuilder = new StringBuilder();
    @Positive
    char[] chars = new char[10];

    // :: error: (argument)
    @Positive
    stringBuilder.append(chars, 5, 7);

    @Positive
    stringBuilder.append(chars, 5, 4);
    @Positive
  }
