// Source-based slice around line 63
// Method: <com.google.common.graph.MapIteratorCache: V put(K,V)>

   * concurrently. For more information, see AbstractNetworkTest.concurrentIteration.
   */
  private transient volatile @Nullable Entry<K, V> cacheEntry;

  MapIteratorCache(Map<K, V> backingMap) {
    this.backingMap = checkNotNull(backingMap);
  }

  @CanIgnoreReturnValue
  final @Nullable V put(K key, V value) {
    checkNotNull(key);
    checkNotNull(value);
    clearCache();
    return backingMap.put(key, value);
  }

  @CanIgnoreReturnValue
  final @Nullable V remove(Object key) {
    checkNotNull(key);
    clearCache();
