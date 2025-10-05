// Source-based slice around line 157
// Method: <com.google.common.math.DoubleMath: long roundToLong(double,RoundingMode)>

   *           mode, is either less than {@code Long.MIN_VALUE} or greater than {@code
   *           Long.MAX_VALUE}
   *       <li>{@code x} is not a mathematical integer and {@code mode} is {@link
   *           RoundingMode#UNNECESSARY}
   *     </ul>
   */
  @GwtIncompatible // #roundIntermediate
  // Whenever both tests are cheap and functional, it's faster to use &, | instead of &&, ||
  @SuppressWarnings("ShortCircuitBoolean")
  public static long roundToLong(double x, RoundingMode mode) {
    double z = roundIntermediate(x, mode);
    checkInRangeForRoundingInputs(
        MIN_LONG_AS_DOUBLE - z < 1.0 & z < MAX_LONG_AS_DOUBLE_PLUS_ONE, x, mode);
    return (long) z;
  }

  private static final double MIN_LONG_AS_DOUBLE = -0x1p63;
  /*
   * We cannot store Long.MAX_VALUE as a double without losing precision. Instead, we store
   * Long.MAX_VALUE + 1 == -Long.MIN_VALUE, and then offset all comparisons by 1.
