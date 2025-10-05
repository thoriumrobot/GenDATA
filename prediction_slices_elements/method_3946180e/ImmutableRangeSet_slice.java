// Source-based slice around line 309
// Method: <com.google.common.collect.ImmutableRangeSet: ImmutableSet asRanges()>

   */
  @Deprecated
  @Override
  @DoNotCall("Always throws UnsupportedOperationException")
  public void removeAll(Iterable<Range<C>> other) {
    throw new UnsupportedOperationException();
  }

  @Override
  public ImmutableSet<Range<C>> asRanges() {
    if (ranges.isEmpty()) {
      return ImmutableSet.of();
    }
    return new RegularImmutableSortedSet<>(ranges, rangeLexOrdering());
  }

  @Override
  public ImmutableSet<Range<C>> asDescendingSetOfRanges() {
    if (ranges.isEmpty()) {
      return ImmutableSet.of();
