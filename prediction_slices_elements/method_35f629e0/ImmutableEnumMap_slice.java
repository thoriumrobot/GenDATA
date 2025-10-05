// Source-based slice around line 82
// Method: <com.google.common.collect.ImmutableEnumMap: V get(Object)>

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
    if (object == this) {
      return true;
    }
    if (object instanceof ImmutableEnumMap) {
      object = ((ImmutableEnumMap<?, ?>) object).delegate;
