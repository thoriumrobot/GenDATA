/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

    @Positive
  void test(@LTLengthOf("array") int x) {
    // :: error: (unary.increment)
    @Positive
    x++;
    // :: error: (unary.increment)
    @Positive
    ++x;
    // :: error: (assignment)
    @Positive
    x = x + 1;
    @Positive
  }
