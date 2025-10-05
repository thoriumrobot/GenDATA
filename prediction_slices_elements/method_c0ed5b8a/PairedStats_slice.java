// Source-based slice around line 77
// Method: <com.google.common.math.PairedStats: Stats yStats()>

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
   *
   * <p>This is guaranteed to return zero if the dataset contains a single pair of finite values. It
   * is not guaranteed to return zero when the dataset consists of the same pair of values multiple
   * times, due to numerical errors.
   *
