/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_logical_expression

    @Positive
  void m2() {
    @Positive
    int[] arr = {1, 2, 3};
    @Positive
    @LTEqLengthOf({"arr"}) int a = Array.getLength(arr);
    @Positive
  }
