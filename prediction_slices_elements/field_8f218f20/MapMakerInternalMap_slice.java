// Source-based slice around line 130
// Method: com.google.common.collect.MapMakerInternalMap.DRAIN_THRESHOLD

  static final int CONTAINS_VALUE_RETRIES = 3;

  /**
   * Number of cache access operations that can be buffered per segment before the cache's recency
   * ordering information is updated. This is used to avoid lock contention by recording a memento
   * of reads and delaying a lock acquisition until the threshold is crossed or a mutation occurs.
   *
   * <p>This must be a (2^n)-1 as it is used as a mask.
   */
  static final int DRAIN_THRESHOLD = 0x3F;

  /**
   * Maximum number of entries to be drained in a single cleanup run. This applies independently to
   * the cleanup queue and both reference queues.
   */
  // TODO(fry): empirically optimize this
  static final int DRAIN_MAX = 16;

  // Fields

