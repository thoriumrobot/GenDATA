// Source-based slice around line 89
// Method: <com.google.common.collect.ImmutableSortedAsList: Spliterator spliterator()>

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
        comparator());
  }

  // redeclare to help optimizers with b/310253115
  @SuppressWarnings("RedundantOverride")
  @Override
