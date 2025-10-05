// Source-based slice around line 165
// Method: <com.google.common.cache.LoadingCache: ConcurrentMap asMap()>

  void refresh(K key);

  /**
   * {@inheritDoc}
   *
   * <p><b>Note that although the view <i>is</i> modifiable, no method on the returned map will ever
   * cause entries to be automatically loaded.</b>
   */
  @Override
  ConcurrentMap<K, V> asMap();
}
