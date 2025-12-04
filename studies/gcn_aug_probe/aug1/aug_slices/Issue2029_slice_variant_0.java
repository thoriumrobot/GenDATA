/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_logical_expression

    @Positive
  void lessThanUpperBound(@NonNegative @LessThan("#2") int index, @NonNegative int size, char val) {
    @Positive
    char[] arr = new char[size];
    @Positive
    arr[index] = val;
    @Positive
  }

    @Positive
  void LessThanOffsetLowerBound(
    @Positive
      int[] array, @NonNegative @LTLengthOf("#1") int n, @NonNegative @LessThan("#2 + 1") int k) {
    @Positive
    array[n - k] = 10;
    @Positive
  }
