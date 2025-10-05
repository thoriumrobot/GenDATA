// Source-based slice around line 82
// Method: <com.google.common.collect.EmptyContiguousSet: ContiguousSet tailSetImpl(C,boolean)>

  }

  @Override
  ContiguousSet<C> subSetImpl(
      C fromElement, boolean fromInclusive, C toElement, boolean toInclusive) {
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
