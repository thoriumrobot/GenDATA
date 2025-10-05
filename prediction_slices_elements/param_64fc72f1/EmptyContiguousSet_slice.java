// Source-based slice around line 66
// Method: <com.google.common.collect.EmptyContiguousSet: Range range(BoundType,BoundType)>

    return this;
  }

  @Override
  public Range<C> range() {
    throw new NoSuchElementException();
  }

  @Override
  public Range<C> range(BoundType lowerBoundType, BoundType upperBoundType) {
    throw new NoSuchElementException();
  }

  @Override
  ContiguousSet<C> headSetImpl(C toElement, boolean inclusive) {
    return this;
  }

  @Override
  ContiguousSet<C> subSetImpl(
