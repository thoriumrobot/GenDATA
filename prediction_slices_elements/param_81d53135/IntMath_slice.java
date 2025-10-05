// Source-based slice around line 264
// Method: <com.google.common.math.IntMath: int sqrt(int,RoundingMode)>

  /**
   * Returns the square root of {@code x}, rounded with the specified rounding mode.
   *
   * @throws IllegalArgumentException if {@code x < 0}
   * @throws ArithmeticException if {@code mode} is {@link RoundingMode#UNNECESSARY} and {@code
   *     sqrt(x)} is not an integer
   */
  @GwtIncompatible // need BigIntegerMath to adequately test
  @SuppressWarnings("fallthrough")
  public static int sqrt(int x, RoundingMode mode) {
    checkNonNegative("x", x);
    int sqrtFloor = sqrtFloor(x);
    switch (mode) {
      case UNNECESSARY:
        checkRoundingUnnecessary(sqrtFloor * sqrtFloor == x); // fall through
      case FLOOR:
      case DOWN:
        return sqrtFloor;
      case CEILING:
      case UP:
