// Source-based slice around line 125
// Method: <com.google.common.collect.RegularContiguousSet: C first()>

    return right != null && Range.compareOrThrow(left, right) == 0;
  }

  @Override
  boolean isPartialView() {
    return false;
  }

  @Override
  public C first() {
    // requireNonNull is safe because we checked the range is not empty in ContiguousSet.create.
    return requireNonNull(range.lowerBound.leastValueAbove(domain));
  }

  @Override
  public C last() {
    // requireNonNull is safe because we checked the range is not empty in ContiguousSet.create.
    return requireNonNull(range.upperBound.greatestValueBelow(domain));
  }

