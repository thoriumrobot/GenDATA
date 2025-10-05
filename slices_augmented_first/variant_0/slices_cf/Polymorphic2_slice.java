    @Positive
  void testSameLen(int @SameLen("array1") [] a, int @SameLen("array2") [] b) {
    @Positive
    int[] x = mergeSameLen(a, b);
    // :: error: (assignment)
    @Positive
    int @SameLen("array1") [] y = mergeSameLen(a, b);
    @Positive
  }
