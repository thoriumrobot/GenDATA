// Source-based slice around line 179
// Method: <com.google.common.collect.TreeRangeSet: void add(Range)>

       * Either both are null or neither is: Either the set is empty, or it's not. But we check both
       * to make the nullness checker happy.
       */
      throw new NoSuchElementException();
    }
    return Range.create(firstEntry.getValue().lowerBound, lastEntry.getValue().upperBound);
  }

  @Override
  public void add(Range<C> rangeToAdd) {
    checkNotNull(rangeToAdd);

    if (rangeToAdd.isEmpty()) {
      return;
    }

    // We will use { } to illustrate ranges currently in the range set, and < >
    // to illustrate rangeToAdd.
    Cut<C> lbToAdd = rangeToAdd.lowerBound;
    Cut<C> ubToAdd = rangeToAdd.upperBound;
