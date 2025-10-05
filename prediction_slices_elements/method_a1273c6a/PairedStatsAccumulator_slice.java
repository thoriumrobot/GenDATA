// Source-based slice around line 100
// Method: <com.google.common.math.PairedStatsAccumulator: long count()>

    yStats.addAll(values.yStats());
  }

  /** Returns an immutable snapshot of the current statistics. */
  public PairedStats snapshot() {
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
