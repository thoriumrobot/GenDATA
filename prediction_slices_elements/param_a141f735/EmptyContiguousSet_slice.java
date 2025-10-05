// Source-based slice around line 87
// Method: <com.google.common.collect.EmptyContiguousSet: boolean contains(Object)>

    return this;
  }

  @Override
  ContiguousSet<C> tailSetImpl(C fromElement, boolean fromInclusive) {
    return this;
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
