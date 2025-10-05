// Source-based slice around line 241
// Method: <com.google.common.math.PairedStatsAccumulator: double ensureInUnitRange(double)>


  private static double ensurePositive(double value) {
    if (value > 0.0) {
      return value;
    } else {
      return Double.MIN_VALUE;
    }
  }

  private static double ensureInUnitRange(double value) {
    return Doubles.constrainToRange(value, -1.0, 1.0);
  }
}
