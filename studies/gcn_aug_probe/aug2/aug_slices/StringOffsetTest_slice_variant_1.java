/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

    @Positive
  public static void OffsetString() {
    @Positive
    char[] chars = new char[10];

    // :: error: (argument)
    @Positive
    String string2 = new String(chars, 5, 7);

    @Positive
    String string3 = new String(chars, 5, 4);
    @Positive
  }
