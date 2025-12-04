/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_logical_expression

    @Positive
  void withConstant(int[] a, @NonNegative int l) {
    @Positive
    if (a.length - l > 10) {
    @Positive
      int x = a[l + 10];
    @Positive
    }
    @Positive
    if (a.length - 10 > l) {
    @Positive
      int x = a[l + 10];
    @Positive
    }
    @Positive
    if (a.length - l >= 10) {
      // :: error: (array.access.unsafe.high)
    @Positive
      int x = a[l + 10];
    @Positive
      int x1 = a[l + 9];
    @Positive
    }
    @Positive
  }

    @Positive
  void withVariable(int[] a, @NonNegative int l, @NonNegative int j, @NonNegative int k) {
    @Positive
    if (a.length - l > j) {
    @Positive
      if (k <= j) {
    @Positive
        int x = a[l + k];
    @Positive
      }
    @Positive
    }
    @Positive
    if (a.length - j > l) {
    @Positive
      if (k <= j) {
    @Positive
        int x = a[l + k];
    @Positive
      }
    @Positive
    }
    @Positive
    if (a.length - j >= l) {
    @Positive
      if (k <= j) {
        // :: error: (array.access.unsafe.high)
    @Positive
        int x = a[l + k];
        // :: error: (array.access.unsafe.low)
    @Positive
        int x1 = a[l + k - 1];
    @Positive
      }
    @Positive
    }
    @Positive
  }
