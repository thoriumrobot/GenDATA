// Source-based slice around line 151
// Method: com.google.common.collect.MapMakerInternalMap.segmentShift

   * Mask value for indexing into segments. The upper bits of a key's hash code are used to choose
   * the segment.
   */
  final transient int segmentMask;

  /**
   * Shift value for indexing within segments. Helps prevent entries that end up in the same segment
   * from also ending up in the same bucket.
   */
  final transient int segmentShift;

  /** The segments, each of which is a specialized hash table. */
  final transient Segment<K, V, E, S>[] segments;

  /** The concurrency level. */
  final int concurrencyLevel;

  /** Strategy for comparing keys. */
  final Equivalence<Object> keyEquivalence;

