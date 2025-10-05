// Source-based slice around line 201
// Method: <com.google.common.collect.ImmutableRangeSet: Range rangeContaining(C)>

            Range::lowerBound,
            otherRange.lowerBound,
            Ordering.natural(),
            ANY_PRESENT,
            NEXT_LOWER);
    return index != -1 && ranges.get(index).encloses(otherRange);
  }

  @Override
  public @Nullable Range<C> rangeContaining(C value) {
    int index =
        SortedLists.binarySearch(
            ranges,
            Range::lowerBound,
            Cut.belowValue(value),
            Ordering.natural(),
            ANY_PRESENT,
            NEXT_LOWER);
    if (index != -1) {
      Range<C> range = ranges.get(index);
