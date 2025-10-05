// Source-based slice around line 93
// Method: <com.google.common.collect.EmptyContiguousSet: int indexOf(Object)>

  }

  @Override
  public boolean contains(@Nullable Object object) {
    return false;
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
