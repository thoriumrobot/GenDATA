// Source-based slice around line 1103
// Method: <com.google.common.collect.MapMakerInternalMap: E copyEntry(E,E)>

    h += (h << 2) + (h << 14);
    return h ^ (h >>> 16);
  }

  /**
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
