// Source-based slice around line 90
// Method: <com.google.common.collect.RegularImmutableSortedMultiset: int count(Object)>

    return isEmpty() ? null : getEntry(0);
  }

  @Override
  public @Nullable Entry<E> lastEntry() {
    return isEmpty() ? null : getEntry(length - 1);
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

