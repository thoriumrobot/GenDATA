// Source-based slice around line 76
// Method: <com.google.common.collect.EmptyContiguousSet: ContiguousSet subSetImpl(C,boolean,C,boolean)>

    throw new NoSuchElementException();
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
