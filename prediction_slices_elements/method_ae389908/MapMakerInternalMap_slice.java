// Source-based slice around line 1152
// Method: <com.google.common.collect.MapMakerInternalMap: V getLiveValue(E)>


  Segment<K, V, E, S> createSegment(int initialCapacity) {
    return entryHelper.newSegment(this, initialCapacity);
  }

  /**
   * Gets the value from an entry. Returns {@code null} if the entry is invalid, partially-collected
   * or computing.
   */
  @Nullable V getLiveValue(E entry) {
    if (entry.getKey() == null) {
      return null;
    }
    return entry.getValue();
  }

  @SuppressWarnings("unchecked")
  final Segment<K, V, E, S>[] newSegmentArray(int ssize) {
    return (Segment<K, V, E, S>[]) new Segment<?, ?, ?, ?>[ssize];
  }
