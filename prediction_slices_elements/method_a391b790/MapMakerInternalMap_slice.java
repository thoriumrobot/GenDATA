// Source-based slice around line 2460
// Method: <com.google.common.collect.MapMakerInternalMap: boolean remove(Object,Object)>

    if (key == null) {
      return null;
    }
    int hash = hash(key);
    return segmentFor(hash).remove(key, hash);
  }

  @CanIgnoreReturnValue
  @Override
  public boolean remove(@Nullable Object key, @Nullable Object value) {
    if (key == null || value == null) {
      return false;
    }
    int hash = hash(key);
    return segmentFor(hash).remove(key, hash, value);
  }

  @CanIgnoreReturnValue
  @Override
  public boolean replace(K key, @Nullable V oldValue, V newValue) {
