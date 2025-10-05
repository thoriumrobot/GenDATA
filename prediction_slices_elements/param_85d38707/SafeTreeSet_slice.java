// Source-based slice around line 73
// Method: <com.google.common.collect.testing.SafeTreeSet: boolean add(E)>


  private SafeTreeSet(NavigableSet<E> delegate) {
    this.delegate = delegate;
    for (E e : this) {
      checkValid(e);
    }
  }

  @Override
  public boolean add(E element) {
    return delegate.add(checkValid(element));
  }

  @Override
  public boolean addAll(Collection<? extends E> collection) {
    for (E e : collection) {
      checkValid(e);
    }
    return delegate.addAll(collection);
  }
