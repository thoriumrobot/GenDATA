// Source-based slice around line 96
// Method: <com.google.common.collect.RegularImmutableSortedMultiset: int size()>

  }

  @Override
  public int count(@Nullable Object element) {
    int index = elementSet.indexOf(element);
    return (index >= 0) ? getCount(index) : 0;
  }

  @Override
  public int size() {
    long size = cumulativeCounts[offset + length] - cumulativeCounts[offset];
    return Ints.saturatedCast(size);
  }

  @Override
  public ImmutableSortedSet<E> elementSet() {
    return elementSet;
  }

  @Override
