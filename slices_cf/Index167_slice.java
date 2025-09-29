    @NonNegative
  static void fn1(int[] arr, @IndexFor("#1") int i) {
    if (i >= 33) {
      // :: error: (argument)
      fn2(arr, i);
    }
    if (i > 33) {
      // :: error: (argument)
      fn2(arr, i);
    }
    if (i != 33) {
      // :: error: (argument)
      fn2(arr, i);
    }
  }

    @Positive
  static void fn2(int[] arr, @NonNegative @LTOMLengthOf("#1") int i) {
    int c = arr[i + 1];
  }
