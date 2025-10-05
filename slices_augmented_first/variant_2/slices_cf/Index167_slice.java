    @Positive
  static void fn1(int[] arr, @IndexFor("#1") int i) {
    @Positive
    if (i >= 33) {
      // :: error: (argument)
    @Positive
      fn2(arr, i);
    @Positive
    }
    @Positive
    if (i > 33) {
      // :: error: (argument)
    @Positive
      fn2(arr, i);
    @Positive
    }
    @Positive
    if (i != 33) {
      // :: error: (argument)
    @Positive
      fn2(arr, i);
    @Positive
    }
    @Positive
  }
