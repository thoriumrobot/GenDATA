// Source-based slice around line 1053
// Method: <com.google.common.cache.CacheBuilder: void checkNonLoadingCache()>

   * @return a cache having the requested features
   * @since 11.0
   */
  public <K1 extends K, V1 extends V> Cache<K1, V1> build() {
    checkWeightWithWeigher();
    checkNonLoadingCache();
    return new LocalCache.LocalManualCache<>(this);
  }

  private void checkNonLoadingCache() {
    checkState(refreshNanos == UNSET_INT, "refreshAfterWrite requires a LoadingCache");
  }

  private void checkWeightWithWeigher() {
    if (weigher == null) {
      checkState(maximumWeight == UNSET_INT, "maximumWeight requires weigher");
    } else {
      if (strictParsing) {
        checkState(maximumWeight != UNSET_INT, "weigher requires maximumWeight");
      } else {
