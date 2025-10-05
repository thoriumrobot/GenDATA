// Source-based slice around line 109
// Method: <com.google.common.collect.EmptyContiguousSet: boolean isPartialView()>

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
    return true;
  }

  @Override
  public ImmutableList<C> asList() {
