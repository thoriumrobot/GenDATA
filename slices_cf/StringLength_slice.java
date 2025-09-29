  void testNewArraySameLen(String s) {
    @Positive
    int @SameLen("s") [] array = new int[s.length()];
    // ::  error: (assignment)
    @Positive
    int @SameLen("s") [] array1 = new int[s.length() + 1];
  }

  void testStringAssignSameLen(String s, String r) {
    @SameLen("s") String t = s;
    // ::  error: (assignment)
    @SameLen("s") String tN = r;
  }

  void testStringLenEqualSameLen(String s, String r) {
    if (s.length() == r.length()) {
      @SameLen("s") String tN = r;
    }
  }

  void testStringEqualSameLen(String s, String r) {
    if (s == r) {
      @SameLen("s") String tN = r;
    }
  }

  void testOffsetRemoval(
      String s,
      String t,
    @Positive
      @LTLengthOf(value = "#1", offset = "#2.length()") int i,
    @Positive
      @LTLengthOf(value = "#2") int j,
      int k) {
    @Positive
    @LTLengthOf("s") int ij = i + j;
    // ::  error: (assignment)
    @Positive
    @LTLengthOf("s") int ik = i + k;
  }

  void testLengthDivide(@MinLen(1) String s) {
    @Positive
    @IndexFor("s") int i = s.length() / 2;
  }

    @NonNegative
  void testAddDivide(@MinLen(1) String s, @IndexFor("#1") int i, @IndexFor("#1") int j) {
    @NonNegative
    @IndexFor("s") int ij = (i + j) / 2;
  }

  void testRandomMultiply(@MinLen(1) String s, Random r) {
    @Positive
    @LTLengthOf("s") int i = (int) (Math.random() * s.length());
    @Positive
    @LTLengthOf("s") int j = (int) (r.nextDouble() * s.length());
  }
