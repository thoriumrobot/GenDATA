// Source-based slice around line 123
// Method: <com.google.common.cache.ForwardingCache: void cleanUp()>

    return delegate().stats();
  }

  @Override
  public ConcurrentMap<K, V> asMap() {
    return delegate().asMap();
  }

  @Override
  public void cleanUp() {
    delegate().cleanUp();
  }

  /**
   * A simplified version of {@link ForwardingCache} where subclasses can pass in an already
   * constructed {@link Cache} as the delegate.
   *
   * @since 10.0
   */
  public abstract static class SimpleForwardingCache<K, V> extends ForwardingCache<K, V> {
