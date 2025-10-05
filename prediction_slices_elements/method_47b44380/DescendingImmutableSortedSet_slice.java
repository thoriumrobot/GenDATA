// Source-based slice around line 43
// Method: <com.google.common.collect.DescendingImmutableSortedSet: int size()>

    this.forward = forward;
  }

  @Override
  public boolean contains(@Nullable Object object) {
    return forward.contains(object);
  }

  @Override
  public int size() {
    return forward.size();
  }

  @Override
  public UnmodifiableIterator<E> iterator() {
    return forward.descendingIterator();
  }

  @Override
  ImmutableSortedSet<E> headSetImpl(E toElement, boolean inclusive) {
