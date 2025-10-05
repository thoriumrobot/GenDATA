// Source-based slice around line 150
// Method: <com.google.common.collect.testing.SafeTreeSet: boolean isEmpty()>

    return new SafeTreeSet<>(delegate.headSet(checkValid(toElement), inclusive));
  }

  @Override
  public @Nullable E higher(E e) {
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
