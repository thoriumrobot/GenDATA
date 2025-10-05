// Source-based slice around line 103
// Method: <com.google.common.math.Stats: Stats of(Iterable)>

    this.max = max;
  }

  /**
   * Returns statistics over a dataset containing the given values.
   *
   * @param values a series of values, which will be converted to {@code double} values (this may
   *     cause loss of precision)
   */
  public static Stats of(Iterable<? extends Number> values) {
    StatsAccumulator accumulator = new StatsAccumulator();
    accumulator.addAll(values);
    return accumulator.snapshot();
  }

  /**
   * Returns statistics over a dataset containing the given values. The iterator will be completely
   * consumed by this method.
   *
   * @param values a series of values, which will be converted to {@code double} values (this may
