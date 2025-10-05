// Source-based slice around line 104
// Method: <com.google.common.collect.EmptyContiguousSet: UnmodifiableIterator descendingIterator()>

  }

  @Override
  public UnmodifiableIterator<C> iterator() {
    return emptyIterator();
  }

  @GwtIncompatible // NavigableSet
  @Override
  public UnmodifiableIterator<C> descendingIterator() {
    return emptyIterator();
  }

  @Override
  boolean isPartialView() {
    return false;
  }

  @Override
  public boolean isEmpty() {
