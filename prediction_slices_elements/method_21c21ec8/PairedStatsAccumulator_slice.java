// Source-based slice around line 110
// Method: <com.google.common.math.PairedStatsAccumulator: Stats yStats()>

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
   *
   * <p>This is guaranteed to return zero if the dataset contains a single pair of finite values. It
   * is not guaranteed to return zero when the dataset consists of the same pair of values multiple
   * times, due to numerical errors.
   *
