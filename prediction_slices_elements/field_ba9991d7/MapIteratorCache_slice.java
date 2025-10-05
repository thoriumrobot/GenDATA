// Source-based slice around line 56
// Method: com.google.common.graph.MapIteratorCache.cacheEntry

  /*
   * Per JDK: "the behavior of a map entry is undefined if the backing map has been modified after
   * the entry was returned by the iterator, except through the setValue operation on the map entry"
   * As such, this field must be cleared before every map mutation.
   *
   * Note about volatile: volatile doesn't make it safe to read from a mutable graph in one thread
   * while writing to it in another. All it does is help with _reading_ from multiple threads
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
