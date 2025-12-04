/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

    @Positive
  void refineNeqLength(int[] array, @IndexOrHigh("#1") int i) {
    // Refines i <= array.length to i < array.length
    @Positive
    if (i != array.length) {
    @Positive
      refineNeqLengthMOne(array, i);
    @Positive
    }
    // No refinement
    @Positive
    if (i != array.length - 1) {
      // :: error: (argument)
    @Positive
      refineNeqLengthMOne(array, i);
    @Positive
    }
    @Positive
  }

    @Positive
  void refineNeqLengthMOne(int[] array, @IndexFor("#1") int i) {
    // Refines i < array.length to i < array.length - 1
    @Positive
    if (i != array.length - 1) {
    @Positive
      refineNeqLengthMTwo(array, i);
      // :: error: (argument)
    @Positive
      refineNeqLengthMThree(array, i);
    @Positive
    }
    @Positive
  }

    @Positive
  void refineNeqLengthMTwo(int[] array, @NonNegative @LTOMLengthOf("#1") int i) {
    // Refines i < array.length - 1 to i < array.length - 2
    @Positive
    if (i != array.length - 2) {
    @Positive
      refineNeqLengthMThree(array, i);
    @Positive
    }
    // No refinement
    @Positive
    if (i != array.length - 1) {
      // :: error: (argument)
    @Positive
      refineNeqLengthMThree(array, i);
    @Positive
    }
    @Positive
  }

    @Positive
  void refineNeqLengthMTwoNonLiteral(
    @Positive
      int[] array,
    @Positive
      @NonNegative @LTOMLengthOf("#1") int i,
    @Positive
      @IntVal(3) int c3,
    @Positive
      @IntVal({2, 3}) int c23) {
    // Refines i < array.length - 1 to i < array.length - 2
    @Positive
    if (i != array.length - (5 - c3)) {
    @Positive
      refineNeqLengthMThree(array, i);
    @Positive
    }
    // No refinement
    @Positive
    if (i != array.length - c23) {
      // :: error: (argument)
    @Positive
      refineNeqLengthMThree(array, i);
    @Positive
    }
    @Positive
  }
