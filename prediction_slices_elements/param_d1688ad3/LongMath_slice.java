// Source-based slice around line 104
// Method: <com.google.common.math.LongMath: int lessThanBranchFree(long,long)>

    return x > 0 & (x & (x - 1)) == 0;
  }

  /**
   * Returns 1 if {@code x < y} as unsigned longs, and 0 otherwise. Assumes that x - y fits into a
   * signed long. The implementation is branch-free, and benchmarks suggest it is measurably faster
   * than the straightforward ternary expression.
   */
  @VisibleForTesting
  static int lessThanBranchFree(long x, long y) {
    // Returns the sign bit of x - y.
    return (int) (~~(x - y) >>> (Long.SIZE - 1));
  }

  /**
   * Returns the base-2 logarithm of {@code x}, rounded according to the specified rounding mode.
   *
   * @throws IllegalArgumentException if {@code x <= 0}
   * @throws ArithmeticException if {@code mode} is {@link RoundingMode#UNNECESSARY} and {@code x}
   *     is not a power of two
