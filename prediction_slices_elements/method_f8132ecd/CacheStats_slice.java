// Source-based slice around line 189
// Method: <com.google.common.cache.CacheStats: long loadExceptionCount()>

   * Returns the number of times {@link Cache} lookup methods threw an exception while loading a new
   * value. This is usually incremented in conjunction with {@code missCount}, though {@code
   * missCount} is also incremented when cache loading completes successfully (see {@link
   * #loadSuccessCount}). Multiple concurrent misses for the same key will result in a single load
   * operation. This may be incremented not in conjunction with {@code missCount} if the load occurs
   * as a result of a refresh or if the cache loader returned more items than was requested. {@code
   * missCount} may also be incremented not in conjunction with this (nor {@link #loadSuccessCount})
   * on calls to {@code getIfPresent}.
   */
  public long loadExceptionCount() {
    return loadExceptionCount;
  }

  /**
   * Returns the ratio of cache loading attempts which threw exceptions. This is defined as {@code
   * loadExceptionCount / (loadSuccessCount + loadExceptionCount)}, or {@code 0.0} when {@code
   * loadSuccessCount + loadExceptionCount == 0}.
   *
   * <p><b>Note:</b> the values of the metrics are undefined in case of overflow (though it is
   * guaranteed not to throw an exception). If you require specific handling, we recommend
