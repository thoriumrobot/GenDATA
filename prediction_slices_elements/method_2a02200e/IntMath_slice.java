// Source-based slice around line 155
// Method: <com.google.common.math.IntMath: int log10(int,RoundingMode)>

  /**
   * Returns the base-10 logarithm of {@code x}, rounded according to the specified rounding mode.
   *
   * @throws IllegalArgumentException if {@code x <= 0}
   * @throws ArithmeticException if {@code mode} is {@link RoundingMode#UNNECESSARY} and {@code x}
   *     is not a power of ten
   */
  @GwtIncompatible // need BigIntegerMath to adequately test
  @SuppressWarnings("fallthrough")
  public static int log10(int x, RoundingMode mode) {
    checkPositive("x", x);
    int logFloor = log10Floor(x);
    int floorPow = powersOf10[logFloor];
    switch (mode) {
      case UNNECESSARY:
        checkRoundingUnnecessary(x == floorPow);
      // fall through
      case FLOOR:
      case DOWN:
        return logFloor;
