// Source-based slice around line 160
// Method: <com.google.common.collect.testing.SafeTreeSet: E last()>

    return delegate.isEmpty();
  }

  @Override
  public Iterator<E> iterator() {
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
