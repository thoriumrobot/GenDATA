// Source-based slice around line 77
// Method: <com.google.common.collect.EmptyContiguousSet: ContiguousSet subSetImpl(C,boolean,C,boolean)>

  }

  @Override
  ContiguousSet<C> headSetImpl(C toElement, boolean inclusive) {
    return this;
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
