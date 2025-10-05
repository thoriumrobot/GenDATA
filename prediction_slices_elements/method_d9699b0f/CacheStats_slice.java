// Source-based slice around line 114
// Method: <com.google.common.cache.CacheStats: long hitCount()>

   * <p><b>Note:</b> the values of the metrics are undefined in case of overflow (though it is
   * guaranteed not to throw an exception). If you require specific handling, we recommend
   * implementing your own stats collector.
   */
  public long requestCount() {
    return saturatedAdd(hitCount, missCount);
  }

  /** Returns the number of times {@link Cache} lookup methods have returned a cached value. */
  public long hitCount() {
    return hitCount;
  }

  /**
   * Returns the ratio of cache requests which were hits. This is defined as {@code hitCount /
   * requestCount}, or {@code 1.0} when {@code requestCount == 0}. Note that {@code hitRate +
   * missRate =~ 1.0}.
   */
  public double hitRate() {
    long requestCount = requestCount();
