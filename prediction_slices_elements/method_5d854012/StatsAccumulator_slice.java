// Source-based slice around line 171
// Method: <com.google.common.math.StatsAccumulator: void addAll(Stats)>

   */
  public void addAll(LongStream values) {
    addAll(values.collect(StatsAccumulator::new, StatsAccumulator::add, StatsAccumulator::addAll));
  }

  /**
   * Adds the given statistics to the dataset, as if the individual values used to compute the
   * statistics had been added directly.
   */
  public void addAll(Stats values) {
    if (values.count() == 0) {
      return;
    }
    merge(values.count(), values.mean(), values.sumOfSquaresOfDeltas(), values.min(), values.max());
  }

  /**
   * Adds the given statistics to the dataset, as if the individual values used to compute the
   * statistics had been added directly.
   *
