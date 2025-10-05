// Source-based slice around line 234
// Method: <com.google.common.cache.CacheStats: long evictionCount()>

  public double averageLoadPenalty() {
    long totalLoadCount = saturatedAdd(loadSuccessCount, loadExceptionCount);
    return (totalLoadCount == 0) ? 0.0 : (double) totalLoadTime / totalLoadCount;
  }

  /**
   * Returns the number of times an entry has been evicted. This count does not include manual
   * {@linkplain Cache#invalidate invalidations}.
   */
  public long evictionCount() {
    return evictionCount;
  }

  /**
   * Returns a new {@code CacheStats} representing the difference between this {@code CacheStats}
   * and {@code other}. Negative values, which aren't supported by {@code CacheStats} will be
   * rounded up to zero.
   */
  public CacheStats minus(CacheStats other) {
    return new CacheStats(
