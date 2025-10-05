// Source-based slice around line 218
// Method: <com.google.common.collect.ImmutableRangeSet: Range span()>

            NEXT_LOWER);
    if (index != -1) {
      Range<C> range = ranges.get(index);
      return range.contains(value) ? range : null;
    }
    return null;
  }

  @Override
  public Range<C> span() {
    if (ranges.isEmpty()) {
      throw new NoSuchElementException();
    }
    return Range.create(ranges.get(0).lowerBound, ranges.get(ranges.size() - 1).upperBound);
  }

  @Override
  public boolean isEmpty() {
    return ranges.isEmpty();
  }
