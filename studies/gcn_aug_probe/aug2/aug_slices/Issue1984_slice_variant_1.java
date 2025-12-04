/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

    @Positive
  public int m(int[] a, @IntRange(from = 0, to = 12) int i) {
    // :: error: (array.access.unsafe.high.range)
    @Positive
    return a[i];
    @Positive
  }
