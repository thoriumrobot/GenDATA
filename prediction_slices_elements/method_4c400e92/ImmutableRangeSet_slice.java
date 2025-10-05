// Source-based slice around line 460
// Method: <com.google.common.collect.ImmutableRangeSet: ImmutableList intersectRanges(Range)>

    RangeSet<C> copy = TreeRangeSet.create(this);
    copy.removeAll(other);
    return copyOf(copy);
  }

  /**
   * Returns a list containing the nonempty intersections of {@code range} with the ranges in this
   * range set.
   */
  private ImmutableList<Range<C>> intersectRanges(Range<C> range) {
    if (ranges.isEmpty() || range.isEmpty()) {
      return ImmutableList.of();
    } else if (range.encloses(span())) {
      return ranges;
    }

    int fromIndex;
    if (range.hasLowerBound()) {
      fromIndex =
          SortedLists.binarySearch(
