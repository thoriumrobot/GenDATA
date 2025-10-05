// Source-based slice around line 155
// Method: <com.google.common.collect.testing.SafeTreeSet: Iterator iterator()>

    return delegate.higher(checkValid(e));
  }

  @Override
  public boolean isEmpty() {
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
