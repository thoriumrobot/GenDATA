// Source-based slice around line 109
// Method: <com.google.common.cache.CacheStats: long requestCount()>


  /**
   * Returns the number of times {@link Cache} lookup methods have returned either a cached or
   * uncached value. This is defined as {@code hitCount + missCount}.
   *
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
