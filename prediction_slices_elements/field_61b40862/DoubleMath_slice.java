// Source-based slice around line 281
// Method: com.google.common.math.DoubleMath.LN_2

        // so log2(x) is never exactly exponent + 0.5.
        increment = (xScaled * xScaled) > 2.0;
        break;
      default:
        throw new AssertionError();
    }
    return increment ? exponent + 1 : exponent;
  }

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
