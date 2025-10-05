// Source-based slice around line 131
// Method: <com.google.common.collect.RegularContiguousSet: C last()>

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

  @Override
  ImmutableList<C> createAsList() {
    if (domain.supportsFastOffset) {
      return new ImmutableAsList<C>() {
        @Override
        ImmutableSortedSet<C> delegateCollection() {
