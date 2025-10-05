// Source-based slice around line 260
// Method: <com.google.common.math.PairedStats: double ensurePositive(double)>

          .add("yStats", yStats)
          .toString();
    }
  }

  double sumOfProductsOfDeltas() {
    return sumOfProductsOfDeltas;
  }

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
