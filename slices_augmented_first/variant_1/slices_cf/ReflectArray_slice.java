    @Positive
  void testNewInstance(int i) {
    // :: error: (argument)
    @Positive
    Array.newInstance(Object.class, i);
    @Positive
    if (i >= 0) {
    @Positive
      Array.newInstance(Object.class, i);
    @Positive
    }
    @Positive
  }
