// Source-based slice around line 165
// Method: <com.google.common.collect.TreeRangeSet: Range span()>

  private @Nullable Range<C> rangeEnclosing(Range<C> range) {
    checkNotNull(range);
    Entry<Cut<C>, Range<C>> floorEntry = rangesByLowerBound.floorEntry(range.lowerBound);
    return (floorEntry != null && floorEntry.getValue().encloses(range))
        ? floorEntry.getValue()
        : null;
  }

  @Override
  public Range<C> span() {
    Entry<Cut<C>, Range<C>> firstEntry = rangesByLowerBound.firstEntry();
    Entry<Cut<C>, Range<C>> lastEntry = rangesByLowerBound.lastEntry();
    if (firstEntry == null || lastEntry == null) {
      /*
       * Either both are null or neither is: Either the set is empty, or it's not. But we check both
       * to make the nullness checker happy.
       */
      throw new NoSuchElementException();
    }
    return Range.create(firstEntry.getValue().lowerBound, lastEntry.getValue().upperBound);
