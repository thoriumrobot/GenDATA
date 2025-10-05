// Source-based slice around line 256
// Method: <com.google.common.math.PairedStats: double sumOfProductsOfDeltas()>

          .toString();
    } else {
      return MoreObjects.toStringHelper(this)
          .add("xStats", xStats)
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
