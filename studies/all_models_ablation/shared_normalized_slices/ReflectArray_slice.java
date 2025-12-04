    @Positive
  void testFor(Object a) {
    @Positive
    for (int i = 0; i < Array.getLength(a); ++i) {
    @Positive
      Array.setInt(a, i, 1 + Array.getInt(a, i));
    @Positive
    }
    @Positive
  }
