// Source-based slice around line 175
// Method: <com.google.common.collect.testing.SafeTreeSet: E pollLast()>

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
    return delegate.remove(checkValid(object));
  }

  @Override
  public boolean removeAll(Collection<?> c) {
