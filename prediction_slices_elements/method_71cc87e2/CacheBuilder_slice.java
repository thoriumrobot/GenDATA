// Source-based slice around line 605
// Method: <com.google.common.cache.CacheBuilder: Weigher getWeigher()>

  long getMaximumWeight() {
    if (expireAfterWriteNanos == 0 || expireAfterAccessNanos == 0) {
      return 0;
    }
    return (weigher == null) ? maximumSize : maximumWeight;
  }

  // Make a safe contravariant cast now so we don't have to do it over and over.
  @SuppressWarnings("unchecked")
  <K1 extends K, V1 extends V> Weigher<K1, V1> getWeigher() {
    return (Weigher<K1, V1>) MoreObjects.firstNonNull(weigher, OneWeigher.INSTANCE);
  }

  /**
   * Specifies that each key (not value) stored in the cache should be wrapped in a {@link
   * WeakReference} (by default, strong references are used).
   *
   * <p><b>Warning:</b> when this method is used, the resulting cache will use identity ({@code ==})
   * comparison to determine equality of keys. Its {@link Cache#asMap} view will therefore
   * technically violate the {@link Map} specification (in the same way that {@link IdentityHashMap}
