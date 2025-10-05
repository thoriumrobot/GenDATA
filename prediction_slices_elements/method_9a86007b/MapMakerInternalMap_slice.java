// Source-based slice around line 1108
// Method: <com.google.common.collect.MapMakerInternalMap: int hash(Object)>

   * This method is a convenience for testing. Code should call {@link Segment#copyEntry} directly.
   */
  // Guarded By Segment.this
  @VisibleForTesting
  E copyEntry(E original, E newNext) {
    int hash = original.getHash();
    return segmentFor(hash).copyEntry(original, newNext);
  }

  int hash(Object key) {
    int h = keyEquivalence.hash(key);
    return rehash(h);
  }

  void reclaimValue(WeakValueReference<K, V, E> valueReference) {
    E entry = valueReference.getEntry();
    int hash = entry.getHash();
    segmentFor(hash).reclaimValue(entry.getKey(), hash, valueReference);
  }

