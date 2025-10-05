// Source-based slice around line 147
// Method: <com.google.common.graph.MapIteratorCache: void clearCache()>

    Entry<K, V> entry = cacheEntry; // store local reference for thread-safety

    // Check cache. We use == on purpose because it's cheaper and a cache miss is ok.
    if (entry != null && entry.getKey() == key) {
      return entry.getValue();
    }
    return null;
  }

  void clearCache() {
    cacheEntry = null;
  }
}
