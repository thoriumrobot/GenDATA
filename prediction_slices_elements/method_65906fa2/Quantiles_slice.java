// Source-based slice around line 476
// Method: <com.google.common.math.Quantiles: double interpolate(double,double,double,double)>

    }
    return false;
  }

  /**
   * Returns a value a fraction {@code (remainder / scale)} of the way between {@code lower} and
   * {@code upper}. Assumes that {@code lower <= upper}. Correctly handles infinities (but not
   * {@code NaN}).
   */
  private static double interpolate(double lower, double upper, double remainder, double scale) {
    if (lower == NEGATIVE_INFINITY) {
      if (upper == POSITIVE_INFINITY) {
        // Return NaN when lower == NEGATIVE_INFINITY and upper == POSITIVE_INFINITY:
        return NaN;
      }
      // Return NEGATIVE_INFINITY when NEGATIVE_INFINITY == lower <= upper < POSITIVE_INFINITY:
      return NEGATIVE_INFINITY;
    }
    if (upper == POSITIVE_INFINITY) {
      // Return POSITIVE_INFINITY when NEGATIVE_INFINITY < lower <= upper == POSITIVE_INFINITY:
