// Source-based slice around line 166
// Method: <com.google.common.math.Stats: Stats of(DoubleStream)>

   * Returns statistics over a dataset containing the given values. The stream will be completely
   * consumed by this method.
   *
   * <p>If you have a {@code Stream<Double>} rather than a {@code DoubleStream}, you should collect
   * the values using {@link #toStats()} instead.
   *
   * @param values a series of values
   * @since 28.2 (but only since 33.4.0 in the Android flavor)
   */
  public static Stats of(DoubleStream values) {
    return values
        .collect(StatsAccumulator::new, StatsAccumulator::add, StatsAccumulator::addAll)
        .snapshot();
  }

  /**
   * Returns statistics over a dataset containing the given values. The stream will be completely
   * consumed by this method.
   *
   * <p>If you have a {@code Stream<Integer>} rather than an {@code IntStream}, you should collect
