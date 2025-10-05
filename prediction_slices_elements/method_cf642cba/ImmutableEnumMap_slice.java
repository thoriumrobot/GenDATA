// Source-based slice around line 72
// Method: <com.google.common.collect.ImmutableEnumMap: int size()>

    return Iterators.unmodifiableIterator(delegate.keySet().iterator());
  }

  @Override
  Spliterator<K> keySpliterator() {
    return delegate.keySet().spliterator();
  }

  @Override
  public int size() {
    return delegate.size();
  }

  @Override
  public boolean containsKey(@Nullable Object key) {
    return delegate.containsKey(key);
  }

  @Override
  public @Nullable V get(@Nullable Object key) {
