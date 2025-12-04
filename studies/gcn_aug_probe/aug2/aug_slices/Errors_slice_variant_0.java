/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_logical_expression

    @Positive
  void test() {
    @Positive
    int[] arr = new int[5];

    // unsafe
    @Positive
    @GTENegativeOne int n1p = -1;
    @Positive
    @LowerBoundUnknown int u = -10;

    // safe
    @Positive
    @NonNegative int nn = 0;
    @Positive
    @Positive int p = 1;

    // :: error: (array.access.unsafe.low)
    @Positive
    int a = arr[n1p];

    // :: error: (array.access.unsafe.low)
    @Positive
    int b = arr[u];

    @Positive
    int c = arr[nn];
    @Positive
    int d = arr[p];
    @Positive
  }
