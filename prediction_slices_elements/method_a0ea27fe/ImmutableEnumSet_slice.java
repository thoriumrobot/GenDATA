// Source-based slice around line 93
// Method: <com.google.common.collect.ImmutableEnumSet: boolean contains(Object)>

    delegate.forEach(action);
  }

  @Override
  public int size() {
    return delegate.size();
  }

  @Override
  public boolean contains(@Nullable Object object) {
    return delegate.contains(object);
  }

  @Override
  public boolean containsAll(Collection<?> collection) {
    if (collection instanceof ImmutableEnumSet<?>) {
      collection = ((ImmutableEnumSet<?>) collection).delegate;
    }
    return delegate.containsAll(collection);
  }
