// Source-based slice around line 263
// Method: <com.google.common.cache.CacheStats: CacheStats plus(CacheStats)>

   * Returns a new {@code CacheStats} representing the sum of this {@code CacheStats} and {@code
   * other}.
   *
   * <p><b>Note:</b> the values of the metrics are undefined in case of overflow (though it is
   * guaranteed not to throw an exception). If you require specific handling, we recommend
   * implementing your own stats collector.
   *
   * @since 11.0
   */
  public CacheStats plus(CacheStats other) {
    return new CacheStats(
        saturatedAdd(hitCount, other.hitCount),
        saturatedAdd(missCount, other.missCount),
        saturatedAdd(loadSuccessCount, other.loadSuccessCount),
        saturatedAdd(loadExceptionCount, other.loadExceptionCount),
        saturatedAdd(totalLoadTime, other.totalLoadTime),
        saturatedAdd(evictionCount, other.evictionCount));
  }

  @Override
