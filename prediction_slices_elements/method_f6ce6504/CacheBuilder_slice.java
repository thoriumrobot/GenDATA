// Source-based slice around line 1047
// Method: <com.google.common.cache.CacheBuilder: Cache build()>

   * <p>Consider {@link #build(CacheLoader)} instead, if it is feasible to implement a {@code
   * CacheLoader}.
   *
   * <p>This method does not alter the state of this {@code CacheBuilder} instance, so it can be
   * invoked again to create multiple independent caches.
   *
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
