// Source-based slice around line 170
// Method: <com.google.common.collect.testing.SafeTreeSet: E pollFirst()>

    return delegate.last();
  }

  @Override
  public @Nullable E lower(E e) {
    return delegate.lower(checkValid(e));
  }

  @Override
  public @Nullable E pollFirst() {
    return delegate.pollFirst();
  }

  @Override
  public @Nullable E pollLast() {
    return delegate.pollLast();
  }

  @Override
  public boolean remove(Object object) {
