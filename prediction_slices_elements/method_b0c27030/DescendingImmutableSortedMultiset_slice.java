// Source-based slice around line 41
// Method: <com.google.common.collect.DescendingImmutableSortedMultiset: Entry firstEntry()>

    this.forward = forward;
  }

  @Override
  public int count(@Nullable Object element) {
    return forward.count(element);
  }

  @Override
  public @Nullable Entry<E> firstEntry() {
    return forward.lastEntry();
  }

  @Override
  public @Nullable Entry<E> lastEntry() {
    return forward.firstEntry();
  }

  @Override
  public int size() {
