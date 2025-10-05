// Source-based slice around line 96
// Method: <com.google.common.math.StatsAccumulator: void addAll(Iterator)>

    }
  }

  /**
   * Adds the given values to the dataset.
   *
   * @param values a series of values, which will be converted to {@code double} values (this may
   *     cause loss of precision)
   */
  public void addAll(Iterator<? extends Number> values) {
    while (values.hasNext()) {
      add(values.next().doubleValue());
    }
  }

  /**
   * Adds the given values to the dataset.
   *
   * @param values a series of values
   */
