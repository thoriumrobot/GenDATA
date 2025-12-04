    @Positive
  void testMinLenSubtractPositive(@MinLen(10) String s) {
    @Positive
    @Positive int i1 = s.length() - 9;
    @Positive
    @NonNegative int i0 = s.length() - 10;
    // ::  error: (assignment)
    @Positive
    @NonNegative int im1 = s.length() - 11;
    @Positive
  }

    @Positive
  void testNewArraySameLen(String s) {
    @Positive
    int @SameLen("s") [] array = new int[s.length()];
    // ::  error: (assignment)
    @Positive
    int @SameLen("s") [] array1 = new int[s.length() + 1];
    @Positive
  }

    @Positive
  void testStringAssignSameLen(String s, String r) {
    @Positive
    @SameLen("s") String t = s;
    // ::  error: (assignment)
    @Positive
    @SameLen("s") String tN = r;
    @Positive
  }

    @Positive
  void testStringLenEqualSameLen(String s, String r) {
    @Positive
    if (s.length() == r.length()) {
    @Positive
      @SameLen("s") String tN = r;
    @Positive
    }
    @Positive
  }

    @Positive
  void testStringEqualSameLen(String s, String r) {
    @Positive
    if (s == r) {
    @Positive
      @SameLen("s") String tN = r;
    @Positive
    }
    @Positive
  }

    @Positive
  void testOffsetRemoval(
    @Positive
      String s,
    @Positive
      String t,
    @Positive
      @LTLengthOf(value = "#1", offset = "#2.length()") int i,
    @Positive
      @LTLengthOf(value = "#2") int j,
    @Positive
      int k) {
    @Positive
    @LTLengthOf("s") int ij = i + j;
    // ::  error: (assignment)
    @Positive
    @LTLengthOf("s") int ik = i + k;
    @Positive
  }
