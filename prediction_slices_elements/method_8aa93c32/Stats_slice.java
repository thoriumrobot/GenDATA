// Source-based slice around line 229
// Method: <com.google.common.math.Stats: long count()>

        (l, r) -> {
          l.addAll(r);
          return l;
        },
        StatsAccumulator::snapshot,
        Collector.Characteristics.UNORDERED);
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
