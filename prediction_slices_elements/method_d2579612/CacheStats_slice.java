// Source-based slice around line 243
// Method: <com.google.common.cache.CacheStats: CacheStats minus(CacheStats)>

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
        max(0, saturatedSubtract(hitCount, other.hitCount)),
        max(0, saturatedSubtract(missCount, other.missCount)),
        max(0, saturatedSubtract(loadSuccessCount, other.loadSuccessCount)),
        max(0, saturatedSubtract(loadExceptionCount, other.loadExceptionCount)),
        max(0, saturatedSubtract(totalLoadTime, other.totalLoadTime)),
        max(0, saturatedSubtract(evictionCount, other.evictionCount)));
  }

  /**
