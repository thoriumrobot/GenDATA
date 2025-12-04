    @Positive
  void test(@BottomVal int x) {
    @Positive
    int[] a = new int[Integer.valueOf(getOneOrTwo())];
    // :: error: (array.length.negative)
    @Positive
    int[] b = new int[Integer.valueOf(x)];
    @Positive
  }
