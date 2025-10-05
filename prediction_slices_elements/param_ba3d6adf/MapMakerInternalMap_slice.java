// Source-based slice around line 1113
// Method: <com.google.common.collect.MapMakerInternalMap: void reclaimValue(WeakValueReference)>

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

  void reclaimKey(E entry) {
    int hash = entry.getHash();
    segmentFor(hash).reclaimKey(entry, hash);
  }

