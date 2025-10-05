// Source-based slice around line 77
// Method: <com.google.common.collect.ImmutableEnumMap: boolean containsKey(Object)>

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
    return delegate.get(key);
  }

  @Override
  public boolean equals(@Nullable Object object) {
