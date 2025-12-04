/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_logical_expression

    @Positive
  public void testMethodInvocation() {
    @Positive
    requiresIndex("012345", 5);
    // :: error: (argument)
    @Positive
    requiresIndex("012345", 6);
    @Positive
  }
