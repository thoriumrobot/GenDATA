// Source-based slice around line 70
// Method: <com.google.common.collect.ForwardingConcurrentMap: boolean replace(K,V,V)>

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
