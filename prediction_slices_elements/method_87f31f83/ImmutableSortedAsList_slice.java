// Source-based slice around line 44
// Method: <com.google.common.collect.ImmutableSortedAsList: Comparator comparator()>

    super(backingSet, backingList);
  }

  @Override
  ImmutableSortedSet<E> delegateCollection() {
    return (ImmutableSortedSet<E>) super.delegateCollection();
  }

  @Override
  public Comparator<? super E> comparator() {
    return delegateCollection().comparator();
  }

  // Override indexOf() and lastIndexOf() to be O(log N) instead of O(N).

  @GwtIncompatible // ImmutableSortedSet.indexOf
  // TODO(cpovirk): consider manual binary search under GWT to preserve O(log N) lookup
  @Override
  public int indexOf(@Nullable Object target) {
    int index = delegateCollection().indexOf(target);
