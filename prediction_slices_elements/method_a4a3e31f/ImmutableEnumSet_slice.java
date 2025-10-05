// Source-based slice around line 106
// Method: <com.google.common.collect.ImmutableEnumSet: boolean isEmpty()>

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

  @Override
  public boolean equals(@Nullable Object object) {
    if (object == this) {
      return true;
    }
    if (object instanceof ImmutableEnumSet) {
      object = ((ImmutableEnumSet<?>) object).delegate;
