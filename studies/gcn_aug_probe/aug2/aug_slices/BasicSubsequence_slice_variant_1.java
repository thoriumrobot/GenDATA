/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

    @Positive
  void test2(@NonNegative @LessThan("y + 1") int x1, int[] a) {
    @Positive
    x = x1;
    // :: error: (to.not.ltel)
    @Positive
    b = a;
    @Positive
  }

    @Positive
  void test3(@NonNegative @LessThan("y") int x1, int[] a) {
    @Positive
    x = x1;
    // :: error: (to.not.ltel)
    @Positive
    b = a;
    @Positive
  }

    @Positive
  void test4(@NonNegative int x1, int[] a) {
    @Positive
    x = x1;
    // :: error: (from.gt.to) :: error: (to.not.ltel)
    @Positive
    b = a;
    @Positive
  }

    @Positive
  void test5(@GTENegativeOne @LessThan("y + 1") int x1, int[] a) {
    @Positive
    x = x1;
    // :: error: (from.not.nonnegative) :: error: (to.not.ltel)
    @Positive
    b = a;
    @Positive
  }
