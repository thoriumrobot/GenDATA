// Source-based slice around line 1119
// Method: <com.google.common.collect.MapMakerInternalMap: void reclaimKey(E)>

    return rehash(h);
  }

  void reclaimValue(WeakValueReference<K, V, E> valueReference) {
    E entry = valueReference.getEntry();
    int hash = entry.getHash();
    segmentFor(hash).reclaimValue(entry.getKey(), hash, valueReference);
  }

  void reclaimKey(E entry) {
    int hash = entry.getHash();
    segmentFor(hash).reclaimKey(entry, hash);
  }

  /**
   * This method is a convenience for testing. Code should call {@link Segment#getLiveValue}
   * instead.
   */
  @VisibleForTesting
  boolean isLiveForTesting(InternalEntry<K, V, ?> entry) {
