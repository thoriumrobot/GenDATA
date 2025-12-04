/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_logical_expression

    @Positive
  void test(int[] arr) {
    // :: error: (array.access.unsafe.low)
    @Positive
    int i = arr[arr.length - 1];

    @Positive
    if (arr.length > 0) {
    @Positive
      int j = arr[arr.length - 1];
    @Positive
    }
    @Positive
  }
