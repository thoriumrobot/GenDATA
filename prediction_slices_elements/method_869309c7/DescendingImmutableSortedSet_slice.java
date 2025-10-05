// Source-based slice around line 48
// Method: <com.google.common.collect.DescendingImmutableSortedSet: UnmodifiableIterator iterator()>

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
    return forward.tailSet(toElement, inclusive).descendingSet();
  }

  @Override
  ImmutableSortedSet<E> subSetImpl(
