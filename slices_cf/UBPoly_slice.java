    @GTENegativeOne
  public static void poly(char[] a, @NonNegative @PolyUpperBound int i) {
    // :: error: (argument)
    access(a, i);
  }

    @Positive
  public static void access(char[] a, @NonNegative @LTLengthOf("#1") int j) {
    char c = a[j];
  }
