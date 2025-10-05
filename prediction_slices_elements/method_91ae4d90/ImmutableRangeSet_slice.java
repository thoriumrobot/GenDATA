// Source-based slice around line 304
// Method: <com.google.common.collect.ImmutableRangeSet: void removeAll(Iterable)>

  /**
   * Guaranteed to throw an exception and leave the {@code RangeSet} unmodified.
   *
   * @throws UnsupportedOperationException always
   * @deprecated Unsupported operation.
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
