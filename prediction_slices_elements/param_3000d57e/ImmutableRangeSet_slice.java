// Source-based slice around line 188
// Method: <com.google.common.collect.ImmutableRangeSet: boolean encloses(Range)>

        && !ranges.get(ceilingIndex).intersection(otherRange).isEmpty()) {
      return true;
    }
    return ceilingIndex > 0
        && ranges.get(ceilingIndex - 1).isConnected(otherRange)
        && !ranges.get(ceilingIndex - 1).intersection(otherRange).isEmpty();
  }

  @Override
  public boolean encloses(Range<C> otherRange) {
    int index =
        SortedLists.binarySearch(
            ranges,
            Range::lowerBound,
            otherRange.lowerBound,
            Ordering.natural(),
            ANY_PRESENT,
            NEXT_LOWER);
    return index != -1 && ranges.get(index).encloses(otherRange);
  }
