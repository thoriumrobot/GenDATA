// Source-based slice around line 137
// Method: com.google.common.collect.MapMakerInternalMap.DRAIN_MAX

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

  /**
   * Mask value for indexing into segments. The upper bits of a key's hash code are used to choose
   * the segment.
   */
  final transient int segmentMask;

  /**
