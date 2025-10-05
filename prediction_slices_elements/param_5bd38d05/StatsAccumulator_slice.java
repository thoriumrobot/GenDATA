// Source-based slice around line 107
// Method: <com.google.common.math.StatsAccumulator: void addAll(double)>

      add(values.next().doubleValue());
    }
  }

  /**
   * Adds the given values to the dataset.
   *
   * @param values a series of values
   */
  public void addAll(double... values) {
    for (double value : values) {
      add(value);
    }
  }

  /**
   * Adds the given values to the dataset.
   *
   * @param values a series of values
   */
