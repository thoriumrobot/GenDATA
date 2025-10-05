// Source-based slice around line 121
// Method: com.google.common.collect.MapMakerInternalMap.CONTAINS_VALUE_RETRIES

   * constructors with arguments. MUST be a power of two no greater than {@code 1<<30} to ensure
   * that entries are indexable using ints.
   */
  static final int MAXIMUM_CAPACITY = Ints.MAX_POWER_OF_TWO;

  /** The maximum number of segments to allow; used to bound constructor arguments. */
  static final int MAX_SEGMENTS = 1 << 16; // slightly conservative

  /** Number of (unsynchronized) retries in the containsValue method. */
  static final int CONTAINS_VALUE_RETRIES = 3;

  /**
   * Number of cache access operations that can be buffered per segment before the cache's recency
   * ordering information is updated. This is used to avoid lock contention by recording a memento
   * of reads and delaying a lock acquisition until the threshold is crossed or a mutation occurs.
   *
   * <p>This must be a (2^n)-1 as it is used as a mask.
   */
  static final int DRAIN_THRESHOLD = 0x3F;

