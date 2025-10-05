// Source-based slice around line 152
// Method: <com.google.common.math.StatsAccumulator: void addAll(IntStream)>

    addAll(values.collect(StatsAccumulator::new, StatsAccumulator::add, StatsAccumulator::addAll));
  }

  /**
   * Adds the given values to the dataset. The stream will be completely consumed by this method.
   *
   * @param values a series of values
   * @since 28.2 (but only since 33.4.0 in the Android flavor)
   */
  public void addAll(IntStream values) {
    addAll(values.collect(StatsAccumulator::new, StatsAccumulator::add, StatsAccumulator::addAll));
  }

  /**
   * Adds the given values to the dataset. The stream will be completely consumed by this method.
   *
   * @param values a series of values, which will be converted to {@code double} values (this may
   *     cause loss of precision for longs of magnitude over 2^53 (slightly over 9e15))
   * @since 28.2 (but only since 33.4.0 in the Android flavor)
   */
