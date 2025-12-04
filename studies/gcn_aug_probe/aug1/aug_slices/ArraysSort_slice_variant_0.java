/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_logical_expression

    @Positive
  void sortInt(int @MinLen(10) [] nums) {
    // Checks the correct handling of the toIndex parameter
    @Positive
    Arrays.sort(nums, 0, 10);
    // :: error: (argument)
    @Positive
    Arrays.sort(nums, 0, 11);
    @Positive
  }
