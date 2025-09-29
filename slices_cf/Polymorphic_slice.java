  void samelen_id(int @SameLen("#2") [] a, int[] a2) {
    int[] banana;
    int @SameLen("a2") [] b = samelen_identity(a);
    // :: error: (assignment)
    int @SameLen("banana") [] c = samelen_identity(b);
  }

  // UpperBound tests
  void ubc_id(
      int[] a,
      int[] b,
    @Positive
      @LTLengthOf("#1") int ai,
    @Positive
      @LTEqLengthOf("#1") int al,
    @Positive
      @LTLengthOf({"#1", "#2"}) int abi,
    @Positive
      @LTEqLengthOf({"#1", "#2"}) int abl) {
    int[] c;

    @Positive
    @LTLengthOf("a") int ai1 = ubc_identity(ai);
    // :: error: (assignment)
    @Positive
    @LTLengthOf("b") int ai2 = ubc_identity(ai);

    @Positive
    @LTEqLengthOf("a") int al1 = ubc_identity(al);
    // :: error: (assignment)
    @Positive
    @LTLengthOf("a") int al2 = ubc_identity(al);

    @Positive
    @LTLengthOf({"a", "b"}) int abi1 = ubc_identity(abi);
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"a", "b", "c"}) int abi2 = ubc_identity(abi);

    @Positive
    @LTEqLengthOf({"a", "b"}) int abl1 = ubc_identity(abl);
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"a", "b", "c"}) int abl2 = ubc_identity(abl);
  }
