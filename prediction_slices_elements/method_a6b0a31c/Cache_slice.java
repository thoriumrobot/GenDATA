// Source-based slice around line 176
// Method: <com.google.common.cache.Cache: ConcurrentMap asMap()>


  /**
   * Returns a view of the entries stored in this cache as a thread-safe map. Modifications made to
   * the map directly affect the cache.
   *
   * <p>Iterators from the returned map are at least <i>weakly consistent</i>: they are safe for
   * concurrent use, but if the cache is modified (including by eviction) after the iterator is
   * created, it is undefined which of the changes (if any) will be reflected in that iterator.
   */
  ConcurrentMap<K, V> asMap();

  /**
   * Performs any pending maintenance operations needed by the cache. Exactly which activities are
   * performed -- if any -- is implementation-dependent.
   */
  void cleanUp();
}
