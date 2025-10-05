// Source-based slice around line 184
// Method: <com.google.common.math.StatsAccumulator: void addAll(StatsAccumulator)>

    merge(values.count(), values.mean(), values.sumOfSquaresOfDeltas(), values.min(), values.max());
  }

  /**
   * Adds the given statistics to the dataset, as if the individual values used to compute the
   * statistics had been added directly.
   *
   * @since 28.2
   */
  public void addAll(StatsAccumulator values) {
    if (values.count() == 0) {
      return;
    }
    merge(values.count(), values.mean(), values.sumOfSquaresOfDeltas(), values.min(), values.max());
  }

  private void merge(
      long otherCount,
      double otherMean,
      double otherSumOfSquaresOfDeltas,
