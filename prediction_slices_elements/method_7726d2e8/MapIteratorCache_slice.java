// Source-based slice around line 93
// Method: <com.google.common.graph.MapIteratorCache: V getWithoutCaching(Object)>

    V value = getIfCached(key);
    // TODO(b/192579700): Use a ternary once it no longer confuses our nullness checker.
    if (value == null) {
      return getWithoutCaching(key);
    } else {
      return value;
    }
  }

  final @Nullable V getWithoutCaching(Object key) {
    checkNotNull(key);
    return backingMap.get(key);
  }

  final boolean containsKey(@Nullable Object key) {
    return getIfCached(key) != null || backingMap.containsKey(key);
  }

  final Set<K> unmodifiableKeySet() {
    return new AbstractSet<K>() {
