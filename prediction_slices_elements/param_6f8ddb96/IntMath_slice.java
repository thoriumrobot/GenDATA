// Source-based slice around line 116
// Method: <com.google.common.math.IntMath: int log2(int,RoundingMode)>

  /**
   * Returns the base-2 logarithm of {@code x}, rounded according to the specified rounding mode.
   *
   * @throws IllegalArgumentException if {@code x <= 0}
   * @throws ArithmeticException if {@code mode} is {@link RoundingMode#UNNECESSARY} and {@code x}
   *     is not a power of two
   */
  @SuppressWarnings("fallthrough")
  // TODO(kevinb): remove after this warning is disabled globally
  public static int log2(int x, RoundingMode mode) {
    checkPositive("x", x);
    switch (mode) {
      case UNNECESSARY:
        checkRoundingUnnecessary(isPowerOfTwo(x));
      // fall through
      case DOWN:
      case FLOOR:
        return (Integer.SIZE - 1) - Integer.numberOfLeadingZeros(x);

      case UP:
