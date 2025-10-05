// Source-based slice around line 317
// Method: <com.google.common.collect.ImmutableRangeSet: ImmutableSet asDescendingSetOfRanges()>

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
    }
    return new RegularImmutableSortedSet<>(ranges.reverse(), Range.<C>rangeLexOrdering().reverse());
  }

  private static final class ComplementRanges<C extends Comparable>
      extends ImmutableList<Range<C>> {

    private final ImmutableList<Range<C>> ranges;
