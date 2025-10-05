// Source-based slice around line 68
// Method: <com.google.common.cache.ForwardingLoadingCache: void refresh(K)>

    return delegate().getAll(keys);
  }

  @Override
  public V apply(K key) {
    return delegate().apply(key);
  }

  @Override
  public void refresh(K key) {
    delegate().refresh(key);
  }

  /**
   * A simplified version of {@link ForwardingLoadingCache} where subclasses can pass in an already
   * constructed {@link LoadingCache} as the delegate.
   *
   * @since 10.0
   */
  public abstract static class SimpleForwardingLoadingCache<K, V>
