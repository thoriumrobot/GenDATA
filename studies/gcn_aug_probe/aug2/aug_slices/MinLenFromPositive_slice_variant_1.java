/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

    @Positive
  void test(@Positive int x) {
    @Positive
    int @MinLen(1) [] y = new int[x];
    @Positive
    @IntRange(from = 1) int z = x;
    @Positive
    @Positive int q = x;
    @Positive
  }

    @Positive
  void foo(int x) {
    @Positive
    test(x);
    @Positive
  }

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
    if (flag) {
    @Positive
      z = x;
    @Positive
    } else {
    @Positive
      z = y;
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
    if (flag) {
    @Positive
      z = x;
    @Positive
    } else {
    @Positive
      z = y;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @Positive int q = z;
    @Positive
    @IntRange(from = -1) int w = z;
    @Positive
  }
