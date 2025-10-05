// Source-based slice around line 163
// Method: <com.google.common.math.StatsAccumulator: void addAll(LongStream)>

  }

  /**
   * Adds the given values to the dataset. The stream will be completely consumed by this method.
   *
   * @param values a series of values, which will be converted to {@code double} values (this may
   *     cause loss of precision for longs of magnitude over 2^53 (slightly over 9e15))
   * @since 28.2 (but only since 33.4.0 in the Android flavor)
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
