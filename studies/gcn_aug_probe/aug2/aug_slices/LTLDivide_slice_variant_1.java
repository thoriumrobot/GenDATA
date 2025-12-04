/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

    @Positive
  int[] test(int[] array) {
    //        @LTLengthOf("array") int len = array.length / 2;
    @Positive
    int len = array.length / 2;
    @Positive
    int[] arr = new int[len];
    @Positive
    for (int a = 0; a < len; a++) {
    @Positive
      arr[a] = array[a];
    @Positive
    }
    @Positive
    return arr;
    @Positive
  }

    @Positive
  void test2(int[] array) {
    @Positive
    int len = array.length;
    @Positive
    int lenM1 = array.length - 1;
    @Positive
    int lenP1 = array.length + 1;
    // :: error: (assignment)
    @Positive
    @LTLengthOf("array") int x = len / 2;
    @Positive
    @LTLengthOf("array") int y = lenM1 / 3;
    @Positive
    @LTEqLengthOf("array") int z = len / 1;
    // :: error: (assignment)
    @Positive
    @LTLengthOf("array") int w = lenP1 / 2;
    @Positive
  }
