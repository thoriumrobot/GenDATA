// Source-based slice around line 1139
// Method: <com.google.common.collect.MapMakerInternalMap: Segment segmentFor(int)>

    return segmentFor(entry.getHash()).getLiveValueForTesting(entry) != null;
  }

  /**
   * Returns the segment that should be used for a key with the given hash.
   *
   * @param hash the hash code for the key
   * @return the segment
   */
  Segment<K, V, E, S> segmentFor(int hash) {
    // TODO(fry): Lazily create segments?
    return segments[(hash >>> segmentShift) & segmentMask];
  }

  Segment<K, V, E, S> createSegment(int initialCapacity) {
    return entryHelper.newSegment(this, initialCapacity);
  }

  /**
   * Gets the value from an entry. Returns {@code null} if the entry is invalid, partially-collected
