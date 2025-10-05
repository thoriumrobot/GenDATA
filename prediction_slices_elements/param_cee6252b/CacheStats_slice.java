// Source-based slice around line 280
// Method: <com.google.common.cache.CacheStats: boolean equals(Object)>

  }

  @Override
  public int hashCode() {
    return Objects.hash(
        hitCount, missCount, loadSuccessCount, loadExceptionCount, totalLoadTime, evictionCount);
  }

  @Override
  public boolean equals(@Nullable Object object) {
    if (object instanceof CacheStats) {
      CacheStats other = (CacheStats) object;
      return hitCount == other.hitCount
          && missCount == other.missCount
          && loadSuccessCount == other.loadSuccessCount
          && loadExceptionCount == other.loadExceptionCount
          && totalLoadTime == other.totalLoadTime
          && evictionCount == other.evictionCount;
    }
    return false;
