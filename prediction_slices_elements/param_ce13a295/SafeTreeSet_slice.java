// Source-based slice around line 233
// Method: <com.google.common.collect.testing.SafeTreeSet: T checkValid(T)>

    return delegate.toArray();
  }

  @Override
  public <T> T[] toArray(T[] a) {
    return delegate.toArray(a);
  }

  @CanIgnoreReturnValue
  private <T> T checkValid(T t) {
    // a ClassCastException is what's supposed to happen!
    @SuppressWarnings("unchecked")
    E e = (E) t;
    int unused = comparator().compare(e, e);
    return t;
  }

  @Override
  public boolean equals(@Nullable Object obj) {
    return delegate.equals(obj);
