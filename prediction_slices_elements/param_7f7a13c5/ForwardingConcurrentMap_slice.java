// Source-based slice around line 51
// Method: <com.google.common.collect.ForwardingConcurrentMap: V putIfAbsent(K,V)>


  /** Constructor for use by subclasses. */
  protected ForwardingConcurrentMap() {}

  @Override
  protected abstract ConcurrentMap<K, V> delegate();

  @CanIgnoreReturnValue
  @Override
  public @Nullable V putIfAbsent(K key, V value) {
    return delegate().putIfAbsent(key, value);
  }

  @CanIgnoreReturnValue
  @Override
  public boolean remove(@Nullable Object key, @Nullable Object value) {
    return delegate().remove(key, value);
  }

  @CanIgnoreReturnValue
