/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_logical_expression

    @Positive
  static void fn1(int[] arr, @IndexFor("#1") int i) {
    @Positive
    if (i >= 33) {
      // :: error: (argument)
    @Positive
      fn2(arr, i);
    @Positive
    }
    @Positive
    if (i > 33) {
      // :: error: (argument)
    @Positive
      fn2(arr, i);
    @Positive
    }
    @Positive
    if (i != 33) {
      // :: error: (argument)
    @Positive
      fn2(arr, i);
    @Positive
    }
    @Positive
  }
