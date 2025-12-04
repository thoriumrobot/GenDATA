/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

    @Positive
  void foo(@IntRange(from = 0, to = 11) int x, int @MinLen(10) [] a) {
    // :: error: (array.access.unsafe.high.range)
    @Positive
    int y = a[x];
    @Positive
  }
