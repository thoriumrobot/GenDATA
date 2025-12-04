/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_logical_expression

    @Positive
  void test() {
    @Positive
    int @MinLen(3) [] arr = {1, 2, 3};
    @Positive
    int i = arr[1];
    // :: error: (array.access.unsafe.high.constant)
    @Positive
    int j = arr[3];
    @Positive
  }
