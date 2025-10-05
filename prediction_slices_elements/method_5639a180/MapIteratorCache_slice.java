// Source-based slice around line 137
// Method: <com.google.common.graph.MapIteratorCache: V getIfCached(Object)>

      @Override
      public boolean contains(@Nullable Object key) {
        return containsKey(key);
      }
    };
  }

  // Internal methods (package-visible, but treat as only subclass-visible)

  @Nullable V getIfCached(@Nullable Object key) {
    Entry<K, V> entry = cacheEntry; // store local reference for thread-safety

    // Check cache. We use == on purpose because it's cheaper and a cache miss is ok.
    if (entry != null && entry.getKey() == key) {
      return entry.getValue();
    }
    return null;
  }

  void clearCache() {
