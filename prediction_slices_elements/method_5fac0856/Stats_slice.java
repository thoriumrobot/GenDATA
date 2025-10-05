// Source-based slice around line 216
// Method: <com.google.common.math.Stats: Collector toStats()>

   * of any type of boxed {@link Number} into a {@link Stats}. Use by calling {@code
   * boxedNumericStream.collect(toStats())}. The numbers will be converted to {@code double} values
   * (which may cause loss of precision).
   *
   * <p>If you have any of the primitive streams {@code DoubleStream}, {@code IntStream}, or {@code
   * LongStream}, you should use the factory method {@link #of} instead.
   *
   * @since 28.2 (but only since 33.4.0 in the Android flavor)
   */
  public static Collector<Number, StatsAccumulator, Stats> toStats() {
    return Collector.of(
        StatsAccumulator::new,
        (a, x) -> a.add(x.doubleValue()),
        (l, r) -> {
          l.addAll(r);
          return l;
        },
        StatsAccumulator::snapshot,
        Collector.Characteristics.UNORDERED);
  }
