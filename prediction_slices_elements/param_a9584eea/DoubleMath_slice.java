// Source-based slice around line 290
// Method: <com.google.common.math.DoubleMath: boolean isMathematicalInteger(double)>

  private static final double LN_2 = log(2);

  /**
   * Returns {@code true} if {@code x} represents a mathematical integer.
   *
   * <p>This is equivalent to, but not necessarily implemented as, the expression {@code
   * !Double.isNaN(x) && !Double.isInfinite(x) && x == Math.rint(x)}.
   */
  @GwtIncompatible // java.lang.Math.getExponent, com.google.common.math.DoubleUtils
  public static boolean isMathematicalInteger(double x) {
    return isFinite(x)
        && (x == 0.0
            || SIGNIFICAND_BITS - Long.numberOfTrailingZeros(getSignificand(x)) <= getExponent(x));
  }

  /**
   * Returns {@code n!}, that is, the product of the first {@code n} positive integers, {@code 1} if
   * {@code n == 0}, or {@code n!}, or {@link Double#POSITIVE_INFINITY} if {@code n! >
   * Double.MAX_VALUE}.
   *
