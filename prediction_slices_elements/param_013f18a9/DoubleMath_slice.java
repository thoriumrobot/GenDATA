// Source-based slice around line 130
// Method: <com.google.common.math.DoubleMath: int roundToInt(double,RoundingMode)>

   *           mode, is either less than {@code Integer.MIN_VALUE} or greater than {@code
   *           Integer.MAX_VALUE}
   *       <li>{@code x} is not a mathematical integer and {@code mode} is {@link
   *           RoundingMode#UNNECESSARY}
   *     </ul>
   */
  @GwtIncompatible // #roundIntermediate
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
