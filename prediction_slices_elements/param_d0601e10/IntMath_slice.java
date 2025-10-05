// Source-based slice around line 91
// Method: <com.google.common.math.IntMath: boolean isPowerOfTwo(int)>


  /**
   * Returns {@code true} if {@code x} represents a power of two.
   *
   * <p>This differs from {@code Integer.bitCount(x) == 1}, because {@code
   * Integer.bitCount(Integer.MIN_VALUE) == 1}, but {@link Integer#MIN_VALUE} is not a power of two.
   */
  // Whenever both tests are cheap and functional, it's faster to use &, | instead of &&, ||
  @SuppressWarnings("ShortCircuitBoolean")
  public static boolean isPowerOfTwo(int x) {
    return x > 0 & (x & (x - 1)) == 0;
  }

  /**
   * Returns 1 if {@code x < y} as unsigned integers, and 0 otherwise. Assumes that x - y fits into
   * a signed int. The implementation is branch-free, and benchmarks suggest it is measurably (if
   * narrowly) faster than the straightforward ternary expression.
   */
  @VisibleForTesting
  static int lessThanBranchFree(int x, int y) {
