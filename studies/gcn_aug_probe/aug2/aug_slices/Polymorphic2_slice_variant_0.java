/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_logical_expression

    @Positive
  int @PolySameLen [] mergeSameLen(int @PolySameLen [] a, int @PolySameLen [] b) {
    @Positive
    return flag ? a : b;
    @Positive
  }

    @Positive
  int[] array1 = new int[2];
    @Positive
  int[] array2 = new int[2];

    @Positive
  void testSameLen(int @SameLen("array1") [] a, int @SameLen("array2") [] b) {
    @Positive
    int[] x = mergeSameLen(a, b);
    // :: error: (assignment)
    @Positive
    int @SameLen("array1") [] y = mergeSameLen(a, b);
    @Positive
  }

    @Positive
  @PolyUpperBound int mergeUpperBound(@PolyUpperBound int a, @PolyUpperBound int b) {
    @Positive
    return flag ? a : b;
    @Positive
  }
