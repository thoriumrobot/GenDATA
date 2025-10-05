// Source-based slice around line 268
// Method: <com.google.common.math.PairedStats: double ensureInUnitRange(double)>


  private static double ensurePositive(double value) {
    if (value > 0.0) {
      return value;
    } else {
      return Double.MIN_VALUE;
    }
  }

  private static double ensureInUnitRange(double value) {
    if (value >= 1.0) {
      return 1.0;
    }
    if (value <= -1.0) {
      return -1.0;
    }
    return value;
  }

  // Serialization helpers
