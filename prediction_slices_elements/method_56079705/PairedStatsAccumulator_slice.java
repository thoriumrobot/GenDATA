// Source-based slice around line 233
// Method: <com.google.common.math.PairedStatsAccumulator: double ensurePositive(double)>

      } else {
        return LinearTransformation.horizontal(yStats.mean());
      }
    } else {
      checkState(yStats.sumOfSquaresOfDeltas() > 0.0);
      return LinearTransformation.vertical(xStats.mean());
    }
  }

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
