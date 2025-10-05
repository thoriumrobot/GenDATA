// Source-based slice around line 137
// Method: com.google.common.math.DoubleMath.MIN_INT_AS_DOUBLE

  // Whenever both tests are cheap and functional, it's faster to use &, | instead of &&, ||
  @SuppressWarnings("ShortCircuitBoolean")
  public static int roundToInt(double x, RoundingMode mode) {
    double z = roundIntermediate(x, mode);
    checkInRangeForRoundingInputs(
        z > MIN_INT_AS_DOUBLE - 1.0 & z < MAX_INT_AS_DOUBLE + 1.0, x, mode);
    return (int) z;
  }

  private static final double MIN_INT_AS_DOUBLE = -0x1p31;
  private static final double MAX_INT_AS_DOUBLE = 0x1p31 - 1.0;

  /**
   * Returns the {@code long} value that is equal to {@code x} rounded with the specified rounding
   * mode, if possible.
   *
   * @throws ArithmeticException if
   *     <ul>
   *       <li>{@code x} is infinite or NaN
   *       <li>{@code x}, after being rounded to a mathematical integer using the specified rounding
