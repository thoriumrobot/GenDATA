// Source-based slice around line 83
// Method: <com.google.common.collect.ImmutableSortedAsList: ImmutableList subListUnchecked(int,int)>

  }

  @GwtIncompatible // super.subListUnchecked does not exist; inherited subList is valid if slow
  /*
   * TODO(cpovirk): if we start to override indexOf/lastIndexOf under GWT, we'll want some way to
   * override subList to return an ImmutableSortedAsList for better performance. Right now, I'm not
   * sure there's any performance hit from our failure to override subListUnchecked under GWT
   */
  @Override
  ImmutableList<E> subListUnchecked(int fromIndex, int toIndex) {
    ImmutableList<E> parentSubList = super.subListUnchecked(fromIndex, toIndex);
    return new RegularImmutableSortedSet<E>(parentSubList, comparator()).asList();
  }

  @Override
  public Spliterator<E> spliterator() {
    return CollectSpliterators.indexed(
        size(),
        ImmutableList.SPLITERATOR_CHARACTERISTICS | Spliterator.SORTED | Spliterator.DISTINCT,
        delegateList()::get,
