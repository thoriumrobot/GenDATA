  void testUpperBound2(@LTLengthOf("array1") int a, @LTEqLengthOf("array1") int b) {
    @LTEqLengthOf("array1") int z = mergeUpperBound(a, b);
    // :: error: (assignment)
    @LTLengthOf("array1") int zz = mergeUpperBound(a, b);
  }
