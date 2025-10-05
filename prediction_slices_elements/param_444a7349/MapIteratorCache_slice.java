// Source-based slice around line 71
// Method: <com.google.common.graph.MapIteratorCache: V remove(Object)>

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
    return backingMap.remove(key);
  }

  final void clear() {
    clearCache();
    backingMap.clear();
  }

