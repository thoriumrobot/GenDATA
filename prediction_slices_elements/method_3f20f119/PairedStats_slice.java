// Source-based slice around line 72
// Method: <com.google.common.math.PairedStats: Stats xStats()>

    this.sumOfProductsOfDeltas = sumOfProductsOfDeltas;
  }

  /** Returns the number of pairs in the dataset. */
  public long count() {
    return xStats.count();
  }

  /** Returns the statistics on the {@code x} values alone. */
  public Stats xStats() {
    return xStats;
  }

  /** Returns the statistics on the {@code y} values alone. */
  public Stats yStats() {
    return yStats;
  }

  /**
   * Returns the population covariance of the values. The count must be non-zero.
