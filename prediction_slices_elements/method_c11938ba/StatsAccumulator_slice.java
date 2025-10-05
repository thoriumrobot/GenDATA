// Source-based slice around line 84
// Method: <com.google.common.math.StatsAccumulator: void addAll(Iterable)>

    }
  }

  /**
   * Adds the given values to the dataset.
   *
   * @param values a series of values, which will be converted to {@code double} values (this may
   *     cause loss of precision)
   */
  public void addAll(Iterable<? extends Number> values) {
    for (Number value : values) {
      add(value.doubleValue());
    }
  }

  /**
   * Adds the given values to the dataset.
   *
   * @param values a series of values, which will be converted to {@code double} values (this may
   *     cause loss of precision)
