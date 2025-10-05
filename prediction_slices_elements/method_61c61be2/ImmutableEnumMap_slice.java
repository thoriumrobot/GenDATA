// Source-based slice around line 67
// Method: <com.google.common.collect.ImmutableEnumMap: Spliterator keySpliterator()>

    checkArgument(!delegate.isEmpty());
  }

  @Override
  UnmodifiableIterator<K> keyIterator() {
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
