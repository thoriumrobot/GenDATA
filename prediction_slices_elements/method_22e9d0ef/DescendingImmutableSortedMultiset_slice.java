// Source-based slice around line 56
// Method: <com.google.common.collect.DescendingImmutableSortedMultiset: ImmutableSortedSet elementSet()>

    return forward.firstEntry();
  }

  @Override
  public int size() {
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
