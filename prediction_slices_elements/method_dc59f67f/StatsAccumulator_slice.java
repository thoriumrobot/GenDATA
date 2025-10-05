// Source-based slice around line 118
// Method: <com.google.common.math.StatsAccumulator: void addAll(int)>

      add(value);
    }
  }

  /**
   * Adds the given values to the dataset.
   *
   * @param values a series of values
   */
  public void addAll(int... values) {
    for (int value : values) {
      add(value);
    }
  }

  /**
   * Adds the given values to the dataset.
   *
   * @param values a series of values, which will be converted to {@code double} values (this may
   *     cause loss of precision for longs of magnitude over 2^53 (slightly over 9e15))
