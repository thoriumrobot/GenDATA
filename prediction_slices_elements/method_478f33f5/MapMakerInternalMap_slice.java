// Source-based slice around line 1160
// Method: <com.google.common.collect.MapMakerInternalMap: Segment[] newSegmentArray(int)>

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

  // Inner Classes

  /**
   * Segments are specialized versions of hash tables. This subclass inherits from ReentrantLock
   * opportunistically, just to simplify some locking and avoid separate construction.
   */
  @SuppressWarnings("serial") // This class is never serialized.
