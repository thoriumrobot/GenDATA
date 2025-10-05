// Source-based slice around line 57
// Method: <com.google.common.collect.ForwardingConcurrentMap: boolean remove(Object,Object)>


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
  @Override
  public @Nullable V replace(K key, V value) {
    return delegate().replace(key, value);
  }

  @CanIgnoreReturnValue
