/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

    @Positive
  void example2(int @MinLen(2) [] a) {
    @Positive
    int j = 2;
    @Positive
    int x = a.length;
    @Positive
    int y = x - j;
    @Positive
    a[y] = 0;
    @Positive
    for (int i = 0; i < y; i++) {
    @Positive
      a[i + j] = 1;
    @Positive
      a[j + i] = 1;
    @Positive
      a[i + 0] = 1;
    @Positive
      a[i - 1] = 1;
      // ::error: (array.access.unsafe.high)
    @Positive
      a[i + 2 + j] = 1;
    @Positive
    }
    @Positive
  }

    @Positive
  void example3(int @MinLen(2) [] a) {
    @Positive
    int j = 2;
    @Positive
    for (int i = 0; i < a.length - 2; i++) {
    @Positive
      a[i + j] = 1;
    @Positive
    }
    @Positive
  }
