// Source-based slice around line 80
// Method: <com.google.common.collect.RegularImmutableSortedMultiset: Entry firstEntry()>

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

  @Override
  public @Nullable Entry<E> lastEntry() {
    return isEmpty() ? null : getEntry(length - 1);
  }

  @Override
  public int count(@Nullable Object element) {
