  @SuppressWarnings("lowerbound")
  void test(int[] a, @Positive int y) {
    @LTLengthOf("a") int x = a.length - 1;
    @LTLengthOf(
        value = {"a", "a"},
        offset = {"0", "y"})
    int z = x - y;
    a[z + y] = 0;
  }
