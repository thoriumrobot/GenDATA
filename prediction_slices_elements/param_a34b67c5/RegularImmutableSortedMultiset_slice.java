// Source-based slice around line 67
// Method: <com.google.common.collect.RegularImmutableSortedMultiset: Entry getEntry(int)>

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
    checkNotNull(action);
    for (int i = 0; i < length; i++) {
      action.accept(elementSet.asList().get(i), getCount(i));
    }
  }
