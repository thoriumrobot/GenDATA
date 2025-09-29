    @Positive
  void testUpperBound(@LTLengthOf("array1") int a, @LTLengthOf("array2") int b) {
    @GTENegativeOne
    int z = mergeUpperBound(a, b);
    // :: error: (assignment)
    @Positive
    @LTLengthOf("array1") int zz = mergeUpperBound(a, b);
  }

    @Positive
  void testUpperBound2(@LTLengthOf("array1") int a, @LTEqLengthOf("array1") int b) {
    @Positive
    @LTEqLengthOf("array1") int z = mergeUpperBound(a, b);
    // :: error: (assignment)
    @Positive
    @LTLengthOf("array1") int zz = mergeUpperBound(a, b);
  }
