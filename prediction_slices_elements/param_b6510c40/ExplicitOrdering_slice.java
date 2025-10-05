// Source-based slice around line 53
// Method: <com.google.common.collect.ExplicitOrdering: boolean equals(Object)>

  private int rank(T value) {
    Integer rank = rankMap.get(value);
    if (rank == null) {
      throw new IncomparableValueException(value);
    }
    return rank;
  }

  @Override
  public boolean equals(@Nullable Object object) {
    if (object instanceof ExplicitOrdering) {
      ExplicitOrdering<?> that = (ExplicitOrdering<?>) object;
      return this.rankMap.equals(that.rankMap);
    }
    return false;
  }

  @Override
  public int hashCode() {
    return rankMap.hashCode();
