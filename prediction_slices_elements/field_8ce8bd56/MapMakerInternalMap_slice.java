// Source-based slice around line 154
// Method: com.google.common.collect.MapMakerInternalMap.segments

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

  /** Strategy for handling entries and segments in a type-safe and efficient manner. */
  final transient InternalEntryHelper<K, V, E, S> entryHelper;

