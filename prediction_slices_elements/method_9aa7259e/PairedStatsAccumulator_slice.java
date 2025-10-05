// Source-based slice around line 105
// Method: <com.google.common.math.PairedStatsAccumulator: Stats xStats()>

    return new PairedStats(xStats.snapshot(), yStats.snapshot(), sumOfProductsOfDeltas);
  }

  /** Returns the number of pairs in the dataset. */
  public long count() {
    return xStats.count();
  }

  /** Returns an immutable snapshot of the statistics on the {@code x} values alone. */
  public Stats xStats() {
    return xStats.snapshot();
  }

  /** Returns an immutable snapshot of the statistics on the {@code y} values alone. */
  public Stats yStats() {
    return yStats.snapshot();
  }

  /**
   * Returns the population covariance of the values. The count must be non-zero.
