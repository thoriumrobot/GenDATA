// Source-based slice around line 499
// Method: <com.google.common.math.Quantiles: double[] longsToDoubles(long[])>

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
      doubles[i] = longs[i];
    }
    return doubles;
  }

  private static double[] intsToDoubles(int[] ints) {
    int len = ints.length;
