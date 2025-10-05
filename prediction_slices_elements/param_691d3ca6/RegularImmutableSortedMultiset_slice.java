// Source-based slice around line 72
// Method: <com.google.common.collect.RegularImmutableSortedMultiset: void forEachEntry(ObjIntConsumer)>

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

  @Override
  public @Nullable Entry<E> firstEntry() {
    return isEmpty() ? null : getEntry(0);
  }
