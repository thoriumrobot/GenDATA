// Source-based slice around line 989
// Method: <com.google.common.cache.CacheBuilder: RemovalListener getRemovalListener()>

    // safely limiting the kinds of caches this can produce
    @SuppressWarnings("unchecked")
    CacheBuilder<K1, V1> me = (CacheBuilder<K1, V1>) this;
    me.removalListener = checkNotNull(listener);
    return me;
  }

  // Make a safe contravariant cast now so we don't have to do it over and over.
  @SuppressWarnings("unchecked")
  <K1 extends K, V1 extends V> RemovalListener<K1, V1> getRemovalListener() {
    return (RemovalListener<K1, V1>)
        MoreObjects.firstNonNull(removalListener, NullListener.INSTANCE);
  }

  /**
   * Enable the accumulation of {@link CacheStats} during the operation of the cache. Without this
   * {@link Cache#stats} will return zero for all statistics. Note that recording stats requires
   * bookkeeping to be performed with each operation, and thus imposes a performance penalty on
   * cache operation.
   *
