// Source-based slice around line 2490
// Method: <com.google.common.collect.MapMakerInternalMap: void clear()>

  @Override
  public @Nullable V replace(K key, V value) {
    checkNotNull(key);
    checkNotNull(value);
    int hash = hash(key);
    return segmentFor(hash).replace(key, hash, value);
  }

  @Override
  public void clear() {
    for (Segment<K, V, E, S> segment : segments) {
      segment.clear();
    }
  }

  @LazyInit transient @Nullable Set<K> keySet;

  @Override
  public Set<K> keySet() {
    Set<K> ks = keySet;
