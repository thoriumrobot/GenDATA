    @Positive
  static int read(
    @Positive
      char[] a,
    @Positive
      @IndexOrHigh("#1") int off,
    @Positive
      @NonNegative @LTLengthOf(value = "#1", offset = "#2 - 1") int len) {
    @Positive
    int sum = 0;
    @Positive
    int i = 0;
        while (i < len) {
            @Positive
      sum += a[i + off];
    @Positive
            i++;
        }
    @Positive
    return sum;
    @Positive
  }
