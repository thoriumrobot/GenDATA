// Source-based slice around line 2482
// Method: <com.google.common.collect.MapMakerInternalMap: V replace(K,V)>

    if (oldValue == null) {
      return false;
    }
    int hash = hash(key);
    return segmentFor(hash).replace(key, hash, oldValue, newValue);
  }

  @CanIgnoreReturnValue
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
