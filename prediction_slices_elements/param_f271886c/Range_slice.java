// Source-based slice around line 440
// Method: <com.google.common.collect.Range: boolean containsAll(Iterable)>

  @Override
  public boolean test(C input) {
    return contains(input);
  }

  /**
   * Returns {@code true} if every element in {@code values} is {@linkplain #contains contained} in
   * this range.
   */
  public boolean containsAll(Iterable<? extends C> values) {
    if (Iterables.isEmpty(values)) {
      return true;
    }

    // this optimizes testing equality of two range-backed sets
    if (values instanceof SortedSet) {
      SortedSet<? extends C> set = (SortedSet<? extends C>) values;
      Comparator<?> comparator = set.comparator();
      if (Ordering.natural().equals(comparator) || comparator == null) {
        return contains(set.first()) && contains(set.last());
