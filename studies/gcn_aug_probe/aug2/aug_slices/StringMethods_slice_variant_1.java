/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

    @Positive
  void testCharAt(String s, int i) {
    // ::  error: (argument)
    @Positive
    s.charAt(i);
    // ::  error: (argument)
    @Positive
    s.codePointAt(i);

    @Positive
    if (i >= 0 && i < s.length()) {
    @Positive
      s.charAt(i);
    @Positive
      s.codePointAt(i);
    @Positive
    }
    @Positive
  }

    @Positive
  void testCodePointBefore(String s) {
    // ::  error: (argument)
    @Positive
    s.codePointBefore(0);

    @Positive
    if (s.length() > 0) {
    @Positive
      s.codePointBefore(s.length());
    @Positive
    }
    @Positive
  }
