/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

    @Positive
  void assignA(int[] d) {
    // :: error: (to.not.ltel)
    @Positive
    a = d;
    @Positive
  }

    @Positive
  void assignB(int[] d) {
    // :: error: (from.gt.to) :: error: (from.not.nonnegative) :: error: (to.not.ltel)
    @Positive
    b = d;
    @Positive
  }
