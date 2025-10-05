// Source-based slice around line 98
// Method: <com.google.common.collect.EmptyContiguousSet: UnmodifiableIterator iterator()>

  }

  @GwtIncompatible // not used by GWT emulation
  @Override
  int indexOf(@Nullable Object target) {
    return -1;
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
