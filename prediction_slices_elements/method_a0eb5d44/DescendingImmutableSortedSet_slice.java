// Source-based slice around line 53
// Method: <com.google.common.collect.DescendingImmutableSortedSet: ImmutableSortedSet headSetImpl(E,boolean)>

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
      E fromElement, boolean fromInclusive, E toElement, boolean toInclusive) {
    return forward.subSet(toElement, toInclusive, fromElement, fromInclusive).descendingSet();
  }

  @Override
