/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

    @Positive
  void testFor(Object a) {
    @Positive
    for (int i = 0; i < Array.getLength(a); ++i) {
    @Positive
      Array.setInt(a, i, 1 + Array.getInt(a, i));
    @Positive
    }
    @Positive
  }
