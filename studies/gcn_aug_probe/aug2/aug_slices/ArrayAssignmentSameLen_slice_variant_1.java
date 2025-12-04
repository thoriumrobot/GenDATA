/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

    @Positive
  ArrayAssignmentSameLen(int[] array, @IndexFor("#1") int index) {
    @Positive
    i_array = array;
    @Positive
    i_index = index;
    @Positive
  }

    @Positive
  void test1(int[] a, int[] b, @LTEqLengthOf("#1") int index) {
    @Positive
    int[] array = a;
    @Positive
        value = {"array", "b"},
    @Positive
        offset = {"0", "-3"})
    // :: error: (assignment)
    @Positive
    int i = index;
    @Positive
  }

    @Positive
  void test2(int[] a, int[] b, @LTLengthOf("#1") int i) {
    @Positive
    int[] c = a;
    // :: error: (assignment)
    @Positive
    @LTLengthOf(value = {"c", "b"}) int x = i;
    @Positive
    @LTLengthOf("c") int y = i;
    @Positive
  }
