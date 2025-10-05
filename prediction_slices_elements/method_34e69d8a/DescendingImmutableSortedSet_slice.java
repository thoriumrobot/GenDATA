// Source-based slice around line 70
// Method: <com.google.common.collect.DescendingImmutableSortedSet: ImmutableSortedSet descendingSet()>

  }

  @Override
  ImmutableSortedSet<E> tailSetImpl(E fromElement, boolean inclusive) {
    return forward.headSet(fromElement, inclusive).descendingSet();
  }

  @Override
  @GwtIncompatible("NavigableSet")
  public ImmutableSortedSet<E> descendingSet() {
    return forward;
  }

  @Override
  @GwtIncompatible("NavigableSet")
  public UnmodifiableIterator<E> descendingIterator() {
    return forward.iterator();
  }

  @Override
