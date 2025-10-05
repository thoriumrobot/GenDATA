// Source-based slice around line 2377
// Method: <com.google.common.collect.MapMakerInternalMap: boolean containsKey(Object)>

  @Nullable E getEntry(@Nullable Object key) {
    if (key == null) {
      return null;
    }
    int hash = hash(key);
    return segmentFor(hash).getEntry(key, hash);
  }

  @Override
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
