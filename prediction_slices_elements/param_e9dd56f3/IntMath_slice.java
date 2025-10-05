// Source-based slice around line 101
// Method: <com.google.common.math.IntMath: int lessThanBranchFree(int,int)>

    return x > 0 & (x & (x - 1)) == 0;
  }

  /**
   * Returns 1 if {@code x < y} as unsigned integers, and 0 otherwise. Assumes that x - y fits into
   * a signed int. The implementation is branch-free, and benchmarks suggest it is measurably (if
   * narrowly) faster than the straightforward ternary expression.
   */
  @VisibleForTesting
  static int lessThanBranchFree(int x, int y) {
    // The double negation is optimized away by normal Java, but is necessary for GWT
    // to make sure bit twiddling works as expected.
    return ~~(x - y) >>> (Integer.SIZE - 1);
  }

  /**
   * Returns the base-2 logarithm of {@code x}, rounded according to the specified rounding mode.
   *
   * @throws IllegalArgumentException if {@code x <= 0}
   * @throws ArithmeticException if {@code mode} is {@link RoundingMode#UNNECESSARY} and {@code x}
