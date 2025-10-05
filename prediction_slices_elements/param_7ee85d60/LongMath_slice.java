// Source-based slice around line 158
// Method: <com.google.common.math.LongMath: int log10(long,RoundingMode)>

   * Returns the base-10 logarithm of {@code x}, rounded according to the specified rounding mode.
   *
   * @throws IllegalArgumentException if {@code x <= 0}
   * @throws ArithmeticException if {@code mode} is {@link RoundingMode#UNNECESSARY} and {@code x}
   *     is not a power of ten
   */
  @GwtIncompatible // TODO
  @SuppressWarnings("fallthrough")
  // TODO(kevinb): remove after this warning is disabled globally
  public static int log10(long x, RoundingMode mode) {
    checkPositive("x", x);
    int logFloor = log10Floor(x);
    long floorPow = powersOf10[logFloor];
    switch (mode) {
      case UNNECESSARY:
        checkRoundingUnnecessary(x == floorPow);
      // fall through
      case FLOOR:
      case DOWN:
        return logFloor;
