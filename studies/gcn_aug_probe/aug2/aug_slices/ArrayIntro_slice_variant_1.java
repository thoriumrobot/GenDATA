/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

    @Positive
  void test() {
    @Positive
    int @MinLen(5) [] arr = new int[5];
    @Positive
    int a = 9;
    @Positive
    a += 5;
    @Positive
    a -= 2;
    @Positive
    int @MinLen(12) [] arr1 = new int[a];
    @Positive
    int @MinLen(3) [] arr2 = {1, 2, 3};
    // :: error: (assignment)
    @Positive
    int @MinLen(4) [] arr3 = {4, 5, 6};
    // :: error: (assignment)
    @Positive
    int @MinLen(7) [] arr4 = new int[4];
    // :: error: (assignment)
    @Positive
    int @MinLen(16) [] arr5 = new int[a];
    @Positive
  }
