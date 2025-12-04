/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_logical_expression

    @Positive
  public static void MinLen(int @MinLen(10) [] arg, int @MinLen(4) [] arg2) {
    @Positive
    int[] arr;
    @Positive
    if (true) {
    @Positive
      arr = arg;
    @Positive
    } else {
    @Positive
      arr = arg2;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    int @MinLen(10) [] res = arr;
    @Positive
    int @MinLen(4) [] res2 = arr;
    // :: error: (assignment)
    @Positive
    int @BottomVal [] res3 = arr;
    @Positive
  }
