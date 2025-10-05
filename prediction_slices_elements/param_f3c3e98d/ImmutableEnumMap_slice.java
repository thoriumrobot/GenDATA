// Source-based slice around line 87
// Method: <com.google.common.collect.ImmutableEnumMap: boolean equals(Object)>

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
    }
    return delegate.equals(object);
  }

  @Override
