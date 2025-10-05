// Source-based slice around line 2442
// Method: <com.google.common.collect.MapMakerInternalMap: void putAll(Map)>

  @Override
  public @Nullable V putIfAbsent(K key, V value) {
    checkNotNull(key);
    checkNotNull(value);
    int hash = hash(key);
    return segmentFor(hash).put(key, hash, value, true);
  }

  @Override
  public void putAll(Map<? extends K, ? extends V> m) {
    for (Entry<? extends K, ? extends V> e : m.entrySet()) {
      put(e.getKey(), e.getValue());
    }
  }

  @CanIgnoreReturnValue
  @Override
  public @Nullable V remove(@Nullable Object key) {
    if (key == null) {
      return null;
