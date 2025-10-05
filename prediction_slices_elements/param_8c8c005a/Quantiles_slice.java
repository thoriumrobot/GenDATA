// Source-based slice around line 508
// Method: <com.google.common.math.Quantiles: double[] intsToDoubles(int[])>

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
    double[] doubles = new double[len];
    for (int i = 0; i < len; i++) {
      doubles[i] = ints[i];
    }
    return doubles;
  }

  /**
   * Performs an in-place selection to find the element which would appear at a given index in a
