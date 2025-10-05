// Source-based slice around line 2316
// Method: <com.google.common.collect.MapMakerInternalMap: boolean isEmpty()>


  @VisibleForTesting
  Equivalence<Object> valueEquivalence() {
    return entryHelper.valueStrength().defaultEquivalence();
  }

  // ConcurrentMap methods

  @Override
  public boolean isEmpty() {
    /*
     * Sum per-segment modCounts to avoid mis-reporting when elements are concurrently added and
     * removed in one segment while checking another, in which case the table was never actually
     * empty at any point. (The sum ensures accuracy up through at least 1<<31 per-segment
     * modifications before recheck.)  Method containsValue() uses similar constructions for
     * stability checks.
     */
    long sum = 0L;
    Segment<K, V, E, S>[] segments = this.segments;
    for (int i = 0; i < segments.length; ++i) {
