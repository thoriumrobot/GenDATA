// Source-based slice around line 95
// Method: <com.google.common.math.PairedStatsAccumulator: PairedStats snapshot()>

          values.sumOfProductsOfDeltas()
              + (values.xStats().mean() - xStats.mean())
                  * (values.yStats().mean() - yStats.mean())
                  * values.count();
    }
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
