// Source-based slice around line 73
// Method: <com.google.common.math.PairedStatsAccumulator: void addAll(PairedStats)>

      sumOfProductsOfDeltas = NaN;
    }
    yStats.add(y);
  }

  /**
   * Adds the given statistics to the dataset, as if the individual values used to compute the
   * statistics had been added directly.
   */
  public void addAll(PairedStats values) {
    if (values.count() == 0) {
      return;
    }

    xStats.addAll(values.xStats());
    if (yStats.count() == 0) {
      sumOfProductsOfDeltas = values.sumOfProductsOfDeltas();
    } else {
      // This is a generalized version of the calculation in add(double, double) above. Note that
      // non-finite inputs will have sumOfProductsOfDeltas = NaN, so non-finite values will result
