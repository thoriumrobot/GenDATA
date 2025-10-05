// Source-based slice around line 138
// Method: <com.google.common.math.Stats: Stats of(int)>

    accumulator.addAll(values);
    return accumulator.snapshot();
  }

  /**
   * Returns statistics over a dataset containing the given values.
   *
   * @param values a series of values
   */
  public static Stats of(int... values) {
    StatsAccumulator accumulator = new StatsAccumulator();
    accumulator.addAll(values);
    return accumulator.snapshot();
  }

  /**
   * Returns statistics over a dataset containing the given values.
   *
   * @param values a series of values, which will be converted to {@code double} values (this may
   *     cause loss of precision for longs of magnitude over 2^53 (slightly over 9e15))
