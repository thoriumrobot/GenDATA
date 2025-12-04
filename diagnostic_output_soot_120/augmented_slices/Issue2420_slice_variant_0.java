/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_logical_expression

    @Positive
  static void str(String argStr) {
    @Positive
    if (argStr.isEmpty()) {
    @Positive
      return;
    @Positive
    }
    @Positive
    if (argStr == "abc") {
    @Positive
      return;
    @Positive
    }
    // :: error: (argument)
    @Positive
    char c = "abc".charAt(argStr.length() - 1);
    // :: error: (argument)
    @Positive
    char c2 = "abc".charAt(argStr.length());
    @Positive
  }
