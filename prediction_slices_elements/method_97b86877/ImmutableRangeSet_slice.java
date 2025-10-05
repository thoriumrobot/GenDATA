// Source-based slice around line 226
// Method: <com.google.common.collect.ImmutableRangeSet: boolean isEmpty()>

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

  /**
   * Guaranteed to throw an exception and leave the {@code RangeSet} unmodified.
   *
   * @throws UnsupportedOperationException always
   * @deprecated Unsupported operation.
   */
  @Deprecated
