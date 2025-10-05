// Source-based slice around line 596
// Method: <com.google.common.cache.CacheBuilder: long getMaximumWeight()>

    }

    // safely limiting the kinds of caches this can produce
    @SuppressWarnings("unchecked")
    CacheBuilder<K1, V1> me = (CacheBuilder<K1, V1>) this;
    me.weigher = checkNotNull(weigher);
    return me;
  }

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
