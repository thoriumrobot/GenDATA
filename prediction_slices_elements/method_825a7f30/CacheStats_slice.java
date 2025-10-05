// Source-based slice around line 274
// Method: <com.google.common.cache.CacheStats: int hashCode()>

        saturatedAdd(hitCount, other.hitCount),
        saturatedAdd(missCount, other.missCount),
        saturatedAdd(loadSuccessCount, other.loadSuccessCount),
        saturatedAdd(loadExceptionCount, other.loadExceptionCount),
        saturatedAdd(totalLoadTime, other.totalLoadTime),
        saturatedAdd(evictionCount, other.evictionCount));
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
