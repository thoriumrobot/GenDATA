// Source-based slice around line 135
// Method: <com.google.common.collect.TreeRangeSet: boolean intersects(Range)>

    if (floorEntry != null && floorEntry.getValue().contains(value)) {
      return floorEntry.getValue();
    } else {
      // TODO(kevinb): revisit this design choice
      return null;
    }
  }

  @Override
  public boolean intersects(Range<C> range) {
    checkNotNull(range);
    Entry<Cut<C>, Range<C>> ceilingEntry = rangesByLowerBound.ceilingEntry(range.lowerBound);
    if (ceilingEntry != null
        && ceilingEntry.getValue().isConnected(range)
        && !ceilingEntry.getValue().intersection(range).isEmpty()) {
      return true;
    }
    Entry<Cut<C>, Range<C>> priorEntry = rangesByLowerBound.lowerEntry(range.lowerBound);
    return priorEntry != null
        && priorEntry.getValue().isConnected(range)
