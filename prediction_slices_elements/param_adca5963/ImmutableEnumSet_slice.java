// Source-based slice around line 98
// Method: <com.google.common.collect.ImmutableEnumSet: boolean containsAll(Collection)>

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

  @Override
  public boolean isEmpty() {
    return delegate.isEmpty();
  }
