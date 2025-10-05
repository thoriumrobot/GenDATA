// Source-based slice around line 199
// Method: <com.google.common.math.Stats: Stats of(LongStream)>

   * consumed by this method.
   *
   * <p>If you have a {@code Stream<Long>} rather than a {@code LongStream}, you should collect the
   * values using {@link #toStats()} instead.
   *
   * @param values a series of values, which will be converted to {@code double} values (this may
   *     cause loss of precision for longs of magnitude over 2^53 (slightly over 9e15))
   * @since 28.2 (but only since 33.4.0 in the Android flavor)
   */
  public static Stats of(LongStream values) {
    return values
        .collect(StatsAccumulator::new, StatsAccumulator::add, StatsAccumulator::addAll)
        .snapshot();
  }

  /**
   * Returns a {@link Collector} which accumulates statistics from a {@link java.util.stream.Stream}
   * of any type of boxed {@link Number} into a {@link Stats}. Use by calling {@code
   * boxedNumericStream.collect(toStats())}. The numbers will be converted to {@code double} values
   * (which may cause loss of precision).
