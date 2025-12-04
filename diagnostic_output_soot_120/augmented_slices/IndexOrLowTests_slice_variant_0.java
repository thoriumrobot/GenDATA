/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_logical_expression

    @Positive
  void test() {

    @Positive
    if (index != -1) {
    @Positive
      array[index] = 1;
    @Positive
    }

    @Positive
    @IndexOrHigh("array") int y = index + 1;
    // :: error: (array.access.unsafe.high)
    @Positive
    array[y] = 1;
    @Positive
    if (y < array.length) {
    @Positive
      array[y] = 1;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    index = array.length;
    @Positive
  }
