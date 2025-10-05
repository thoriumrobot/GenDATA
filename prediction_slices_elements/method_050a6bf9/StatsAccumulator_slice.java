// Source-based slice around line 225
// Method: <com.google.common.math.StatsAccumulator: long count()>

    }
  }

  /** Returns an immutable snapshot of the current statistics. */
  public Stats snapshot() {
    return new Stats(count, mean, sumOfSquaresOfDeltas, min, max);
  }

  /** Returns the number of values. */
  public long count() {
    return count;
  }

  /**
   * Returns the <a href="http://en.wikipedia.org/wiki/Arithmetic_mean">arithmetic mean</a> of the
   * values. The count must be non-zero.
   *
   * <p>If these values are a sample drawn from a population, this is also an unbiased estimator of
   * the arithmetic mean of the population.
   *
