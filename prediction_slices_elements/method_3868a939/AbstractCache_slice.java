// Source-based slice around line 136
// Method: <com.google.common.cache.AbstractCache: ConcurrentMap asMap()>

    throw new UnsupportedOperationException();
  }

  @Override
  public CacheStats stats() {
    throw new UnsupportedOperationException();
  }

  @Override
  public ConcurrentMap<K, V> asMap() {
    throw new UnsupportedOperationException();
  }

  /**
   * Accumulates statistics during the operation of a {@link Cache} for presentation by {@link
   * Cache#stats}. This is solely intended for consumption by {@code Cache} implementors.
   *
   * @since 10.0
   */
  public interface StatsCounter {
