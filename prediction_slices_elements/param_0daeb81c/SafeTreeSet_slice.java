// Source-based slice around line 165
// Method: <com.google.common.collect.testing.SafeTreeSet: E lower(E)>

    return delegate.iterator();
  }

  @Override
  public E last() {
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
