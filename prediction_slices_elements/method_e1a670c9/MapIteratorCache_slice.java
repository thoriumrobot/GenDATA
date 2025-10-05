// Source-based slice around line 77
// Method: <com.google.common.graph.MapIteratorCache: void clear()>

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

  @Nullable V get(Object key) {
    checkNotNull(key);
    V value = getIfCached(key);
    // TODO(b/192579700): Use a ternary once it no longer confuses our nullness checker.
    if (value == null) {
      return getWithoutCaching(key);
