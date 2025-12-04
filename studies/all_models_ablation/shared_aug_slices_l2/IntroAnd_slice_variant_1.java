/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

    @Positive
  void test() {
    @Positive
    @NonNegative int a = 1 & 0;
    @Positive
    @NonNegative int b = a & 5;

    // :: error: (assignment)
    @Positive
    @Positive int c = a & b;
    @Positive
    @NonNegative int d = a & b;
    @Positive
    @NonNegative int e = b & a;
    @Positive
  }

    @Positive
  void test_ubc_and(
    @Positive
      @IndexFor("#2") int i, int[] a, @LTLengthOf("#2") int j, int k, @NonNegative int m) {
    @Positive
    int x = a[i & k];
    @Positive
    int x1 = a[k & i];
    // :: error: (array.access.unsafe.low) :: error: (array.access.unsafe.high)
    @Positive
    int y = a[j & k];
    @Positive
    if (j > -1) {
    @Positive
      int z = a[j & k];
    @Positive
    }
    // :: error: (array.access.unsafe.high)
    @Positive
    int w = a[m & k];
    @Positive
    if (m < a.length) {
    @Positive
      int u = a[m & k];
    @Positive
    }
    @Positive
  }
