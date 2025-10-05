// Source-based slice around line 2450
// Method: <com.google.common.collect.MapMakerInternalMap: V remove(Object)>

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
    }
    int hash = hash(key);
    return segmentFor(hash).remove(key, hash);
  }

  @CanIgnoreReturnValue
  @Override
  public boolean remove(@Nullable Object key, @Nullable Object value) {
