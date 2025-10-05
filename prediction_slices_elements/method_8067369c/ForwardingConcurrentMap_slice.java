// Source-based slice around line 63
// Method: <com.google.common.collect.ForwardingConcurrentMap: V replace(K,V)>


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
  @Override
  @SuppressWarnings("nullness") // https://github.com/jspecify/jdk/issues/118
  public boolean replace(K key, V oldValue, V newValue) {
    return delegate().replace(key, oldValue, newValue);
  }
}
