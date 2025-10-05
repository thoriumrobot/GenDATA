// Source-based slice around line 2470
// Method: <com.google.common.collect.MapMakerInternalMap: boolean replace(K,V,V)>

    if (key == null || value == null) {
      return false;
    }
    int hash = hash(key);
    return segmentFor(hash).remove(key, hash, value);
  }

  @CanIgnoreReturnValue
  @Override
  public boolean replace(K key, @Nullable V oldValue, V newValue) {
    checkNotNull(key);
    checkNotNull(newValue);
    if (oldValue == null) {
      return false;
    }
    int hash = hash(key);
    return segmentFor(hash).replace(key, hash, oldValue, newValue);
  }

  @CanIgnoreReturnValue
