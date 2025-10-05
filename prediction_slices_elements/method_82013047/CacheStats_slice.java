// Source-based slice around line 202
// Method: <com.google.common.cache.CacheStats: double loadExceptionRate()>

  /**
   * Returns the ratio of cache loading attempts which threw exceptions. This is defined as {@code
   * loadExceptionCount / (loadSuccessCount + loadExceptionCount)}, or {@code 0.0} when {@code
   * loadSuccessCount + loadExceptionCount == 0}.
   *
   * <p><b>Note:</b> the values of the metrics are undefined in case of overflow (though it is
   * guaranteed not to throw an exception). If you require specific handling, we recommend
   * implementing your own stats collector.
   */
  public double loadExceptionRate() {
    long totalLoadCount = saturatedAdd(loadSuccessCount, loadExceptionCount);
    return (totalLoadCount == 0) ? 0.0 : (double) loadExceptionCount / totalLoadCount;
  }

  /**
   * Returns the total number of nanoseconds the cache has spent loading new values. This can be
   * used to calculate the miss penalty. This value is increased every time {@code loadSuccessCount}
   * or {@code loadExceptionCount} is incremented.
   */
  @SuppressWarnings("GoodTime") // should return a java.time.Duration
