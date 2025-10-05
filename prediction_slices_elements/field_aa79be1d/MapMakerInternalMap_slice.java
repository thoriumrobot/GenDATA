// Source-based slice around line 163
// Method: com.google.common.collect.MapMakerInternalMap.entryHelper

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
  private MapMakerInternalMap(MapMaker builder, InternalEntryHelper<K, V, E, S> entryHelper) {
    concurrencyLevel = min(builder.getConcurrencyLevel(), MAX_SEGMENTS);

    keyEquivalence = builder.getKeyEquivalence();
    this.entryHelper = entryHelper;

