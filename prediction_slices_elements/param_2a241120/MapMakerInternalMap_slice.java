// Source-based slice around line 2386
// Method: <com.google.common.collect.MapMakerInternalMap: boolean containsValue(Object)>

  public boolean containsKey(@Nullable Object key) {
    if (key == null) {
      return false;
    }
    int hash = hash(key);
    return segmentFor(hash).containsKey(key, hash);
  }

  @Override
  public boolean containsValue(@Nullable Object value) {
    if (value == null) {
      return false;
    }

    // This implementation is patterned after ConcurrentHashMap, but without the locking. The only
    // way for it to return a false negative would be for the target value to jump around in the map
    // such that none of the subsequent iterations observed it, despite the fact that at every point
    // in time it was present somewhere int the map. This becomes increasingly unlikely as
    // CONTAINS_VALUE_RETRIES increases, though without locking it is theoretically possible.
    Segment<K, V, E, S>[] segments = this.segments;
