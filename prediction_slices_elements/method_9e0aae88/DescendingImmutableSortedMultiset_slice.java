// Source-based slice around line 51
// Method: <com.google.common.collect.DescendingImmutableSortedMultiset: int size()>

    return forward.lastEntry();
  }

  @Override
  public @Nullable Entry<E> lastEntry() {
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
