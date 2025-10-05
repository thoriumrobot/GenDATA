// Source-based slice around line 142
// Method: <com.google.common.math.StatsAccumulator: void addAll(DoubleStream)>

    }
  }

  /**
   * Adds the given values to the dataset. The stream will be completely consumed by this method.
   *
   * @param values a series of values
   * @since 28.2 (but only since 33.4.0 in the Android flavor)
   */
  public void addAll(DoubleStream values) {
    addAll(values.collect(StatsAccumulator::new, StatsAccumulator::add, StatsAccumulator::addAll));
  }

  /**
   * Adds the given values to the dataset. The stream will be completely consumed by this method.
   *
   * @param values a series of values
   * @since 28.2 (but only since 33.4.0 in the Android flavor)
   */
  public void addAll(IntStream values) {
