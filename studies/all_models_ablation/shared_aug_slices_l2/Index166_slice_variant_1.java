/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

    @Positive
  public void testMethodInvocation() {
    @Positive
    requiresIndex("012345", 5);
    // :: error: (argument)
    @Positive
    requiresIndex("012345", 6);
    @Positive
  }
