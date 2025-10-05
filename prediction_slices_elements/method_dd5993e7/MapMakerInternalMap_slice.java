// Source-based slice around line 2368
// Method: <com.google.common.collect.MapMakerInternalMap: E getEntry(Object)>

    }
    int hash = hash(key);
    return segmentFor(hash).get(key, hash);
  }

  /**
   * Returns the internal entry for the specified key. The entry may be computing or partially
   * collected. Does not impact recency ordering.
   */
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
