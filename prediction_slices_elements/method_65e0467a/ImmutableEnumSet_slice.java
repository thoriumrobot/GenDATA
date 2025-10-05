// Source-based slice around line 111
// Method: <com.google.common.collect.ImmutableEnumSet: boolean equals(Object)>

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
    }
    return delegate.equals(object);
  }

  @Override
