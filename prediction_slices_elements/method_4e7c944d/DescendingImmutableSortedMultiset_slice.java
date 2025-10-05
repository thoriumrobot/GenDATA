// Source-based slice around line 61
// Method: <com.google.common.collect.DescendingImmutableSortedMultiset: Entry getEntry(int)>

    return forward.size();
  }

  @Override
  public ImmutableSortedSet<E> elementSet() {
    return forward.elementSet().descendingSet();
  }

  @Override
  Entry<E> getEntry(int index) {
    return forward.entrySet().asList().reverse().get(index);
  }

  @Override
  public ImmutableSortedMultiset<E> descendingMultiset() {
    return forward;
  }

  @Override
  public ImmutableSortedMultiset<E> headMultiset(E upperBound, BoundType boundType) {
