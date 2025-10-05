    @Positive
  void foo2(int x) {
    // :: error: (argument)
    @Positive
    test(x);
    @Positive
  }

    @Positive
  void test_lub1(boolean flag, @Positive int x, @IntRange(from = 6, to = 25) int y) {
    @Positive
    int z;
    @Positive
    if (!(flag)) {
            @Positive
      z = y;
    @Positive
        } else {
            @Positive
      z = x;
    @Positive
        }
    @Positive
    @Positive int q = z;
    @Positive
    @IntRange(from = 1) int w = z;
    @Positive
  }

    @Positive
  void test_lub2(boolean flag, @Positive int x, @IntRange(from = -1, to = 11) int y) {
    @Positive
    int z;
    @Positive
    if (!(flag)) {
            @Positive
      z = y;
    @Positive
        } else {
            @Positive
      z = x;
    @Positive
        }
    // :: error: (assignment)
    @Positive
    @Positive int q = z;
    @Positive
    @IntRange(from = -1) int w = z;
    @Positive
  }
