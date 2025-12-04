/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

    @Positive
  void pre1(int[] args) {
    @Positive
    int ii = 0;
    @Positive
    while ((ii < args.length)) {
      // :: error: (array.access.unsafe.high)
    @Positive
      int m = args[++ii];
    @Positive
    }
    @Positive
  }
