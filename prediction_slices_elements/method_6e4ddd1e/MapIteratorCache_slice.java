// Source-based slice around line 102
// Method: <com.google.common.graph.MapIteratorCache: Set unmodifiableKeySet()>

  final @Nullable V getWithoutCaching(Object key) {
    checkNotNull(key);
    return backingMap.get(key);
  }

  final boolean containsKey(@Nullable Object key) {
    return getIfCached(key) != null || backingMap.containsKey(key);
  }

  final Set<K> unmodifiableKeySet() {
    return new AbstractSet<K>() {
      @Override
      public UnmodifiableIterator<K> iterator() {
        Iterator<Entry<K, V>> entryIterator = backingMap.entrySet().iterator();

        return new UnmodifiableIterator<K>() {
          @Override
          public boolean hasNext() {
            return entryIterator.hasNext();
          }
