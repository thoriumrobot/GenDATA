// Source-based slice around line 62
// Method: <com.google.common.collect.RegularImmutableSortedMultiset: int getCount(int)>


  RegularImmutableSortedMultiset(
      RegularImmutableSortedSet<E> elementSet, long[] cumulativeCounts, int offset, int length) {
    this.elementSet = elementSet;
    this.cumulativeCounts = cumulativeCounts;
    this.offset = offset;
    this.length = length;
  }

  private int getCount(int index) {
    return (int) (cumulativeCounts[offset + index + 1] - cumulativeCounts[offset + index]);
  }

  @Override
  Entry<E> getEntry(int index) {
    return Multisets.immutableEntry(elementSet.asList().get(index), getCount(index));
  }

  @Override
  public void forEachEntry(ObjIntConsumer<? super E> action) {
