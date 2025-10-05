// Source-based slice around line 94
// Method: <com.google.common.math.LongMath: boolean isPowerOfTwo(long)>


  /**
   * Returns {@code true} if {@code x} represents a power of two.
   *
   * <p>This differs from {@code Long.bitCount(x) == 1}, because {@code
   * Long.bitCount(Long.MIN_VALUE) == 1}, but {@link Long#MIN_VALUE} is not a power of two.
   */
  // Whenever both tests are cheap and functional, it's faster to use &, | instead of &&, ||
  @SuppressWarnings("ShortCircuitBoolean")
  public static boolean isPowerOfTwo(long x) {
    return x > 0 & (x & (x - 1)) == 0;
  }

  /**
   * Returns 1 if {@code x < y} as unsigned longs, and 0 otherwise. Assumes that x - y fits into a
   * signed long. The implementation is branch-free, and benchmarks suggest it is measurably faster
   * than the straightforward ternary expression.
   */
  @VisibleForTesting
  static int lessThanBranchFree(long x, long y) {
