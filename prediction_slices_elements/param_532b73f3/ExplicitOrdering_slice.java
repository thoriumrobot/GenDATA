// Source-based slice around line 40
// Method: <com.google.common.collect.ExplicitOrdering: int compare(T,T)>

  ExplicitOrdering(List<T> valuesInOrder) {
    this(Maps.indexMap(valuesInOrder));
  }

  ExplicitOrdering(ImmutableMap<T, Integer> rankMap) {
    this.rankMap = rankMap;
  }

  @Override
  public int compare(T left, T right) {
    return rank(left) - rank(right); // safe because both are nonnegative
  }

  private int rank(T value) {
    Integer rank = rankMap.get(value);
    if (rank == null) {
      throw new IncomparableValueException(value);
    }
    return rank;
  }
