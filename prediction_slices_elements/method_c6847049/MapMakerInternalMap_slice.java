// Source-based slice around line 1129
// Method: <com.google.common.collect.MapMakerInternalMap: boolean isLiveForTesting(InternalEntry)>

    int hash = entry.getHash();
    segmentFor(hash).reclaimKey(entry, hash);
  }

  /**
   * This method is a convenience for testing. Code should call {@link Segment#getLiveValue}
   * instead.
   */
  @VisibleForTesting
  boolean isLiveForTesting(InternalEntry<K, V, ?> entry) {
    return segmentFor(entry.getHash()).getLiveValueForTesting(entry) != null;
  }

  /**
   * Returns the segment that should be used for a key with the given hash.
   *
   * @param hash the hash code for the key
   * @return the segment
   */
  Segment<K, V, E, S> segmentFor(int hash) {
