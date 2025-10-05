// Source-based slice around line 492
// Method: <com.google.common.math.Quantiles: void checkIndex(int,int)>

      return NEGATIVE_INFINITY;
    }
    if (upper == POSITIVE_INFINITY) {
      // Return POSITIVE_INFINITY when NEGATIVE_INFINITY < lower <= upper == POSITIVE_INFINITY:
      return POSITIVE_INFINITY;
    }
    return lower + (upper - lower) * remainder / scale;
  }

  private static void checkIndex(int index, int scale) {
    if (index < 0 || index > scale) {
      throw new IllegalArgumentException(
          "Quantile indexes must be between 0 and the scale, which is " + scale);
    }
  }

  private static double[] longsToDoubles(long[] longs) {
    int len = longs.length;
    double[] doubles = new double[len];
    for (int i = 0; i < len; i++) {
