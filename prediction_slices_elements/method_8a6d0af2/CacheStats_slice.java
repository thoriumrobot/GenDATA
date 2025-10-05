// Source-based slice around line 225
// Method: <com.google.common.cache.CacheStats: double averageLoadPenalty()>


  /**
   * Returns the average time spent loading new values. This is defined as {@code totalLoadTime /
   * (loadSuccessCount + loadExceptionCount)}.
   *
   * <p><b>Note:</b> the values of the metrics are undefined in case of overflow (though it is
   * guaranteed not to throw an exception). If you require specific handling, we recommend
   * implementing your own stats collector.
   */
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
