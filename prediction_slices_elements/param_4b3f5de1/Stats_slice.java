// Source-based slice around line 150
// Method: <com.google.common.math.Stats: Stats of(long)>

    return accumulator.snapshot();
  }

  /**
   * Returns statistics over a dataset containing the given values.
   *
   * @param values a series of values, which will be converted to {@code double} values (this may
   *     cause loss of precision for longs of magnitude over 2^53 (slightly over 9e15))
   */
  public static Stats of(long... values) {
    StatsAccumulator accumulator = new StatsAccumulator();
    accumulator.addAll(values);
    return accumulator.snapshot();
  }

  /**
   * Returns statistics over a dataset containing the given values. The stream will be completely
   * consumed by this method.
   *
   * <p>If you have a {@code Stream<Double>} rather than a {@code DoubleStream}, you should collect
