// Source-based slice around line 157
// Method: com.google.common.collect.MapMakerInternalMap.concurrencyLevel

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

  /**
   * Creates a new, empty map with the specified strategy, initial capacity and concurrency level.
   */
