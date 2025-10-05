// Source-based slice around line 137
// Method: <com.google.common.collect.RegularContiguousSet: ImmutableList createAsList()>

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
          return RegularContiguousSet.this;
        }

        @Override
        public C get(int i) {
          checkElementIndex(i, size());
