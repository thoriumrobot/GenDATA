// Source-based slice around line 161
// Method: <com.google.common.cache.CacheStats: long loadCount()>

  /**
   * Returns the total number of times that {@link Cache} lookup methods attempted to load new
   * values. This includes both successful load operations and those that threw exceptions. This is
   * defined as {@code loadSuccessCount + loadExceptionCount}.
   *
   * <p><b>Note:</b> the values of the metrics are undefined in case of overflow (though it is
   * guaranteed not to throw an exception). If you require specific handling, we recommend
   * implementing your own stats collector.
   */
  public long loadCount() {
    return saturatedAdd(loadSuccessCount, loadExceptionCount);
  }

  /**
   * Returns the number of times {@link Cache} lookup methods have successfully loaded a new value.
   * This is usually incremented in conjunction with {@link #missCount}, though {@code missCount} is
   * also incremented when an exception is encountered during cache loading (see {@link
   * #loadExceptionCount}). Multiple concurrent misses for the same key will result in a single load
   * operation. This may be incremented not in conjunction with {@code missCount} if the load occurs
   * as a result of a refresh or if the cache loader returned more items than was requested. {@code
