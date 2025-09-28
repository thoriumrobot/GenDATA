  void negative(Object[] a, @LTLengthOf(value = "#1", offset = "100") int i) {
    // Not true for some negative i
    // :: error: (assignment)
    @LTLengthOf(value = "#1", offset = "100") int q = i >> 2;
  }
