// Source-based slice around line 46
// Method: <com.google.common.collect.DescendingImmutableSortedMultiset: Entry lastEntry()>

    return forward.count(element);
  }

  @Override
  public @Nullable Entry<E> firstEntry() {
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
