// Source-based slice around line 67
// Method: <com.google.common.math.PairedStats: long count()>

   * </ul>
   */
  PairedStats(Stats xStats, Stats yStats, double sumOfProductsOfDeltas) {
    this.xStats = xStats;
    this.yStats = yStats;
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
