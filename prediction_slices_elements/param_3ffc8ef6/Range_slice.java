// Source-based slice around line 304
// Method: <com.google.common.collect.Range: Range encloseAll(Iterable)>

  /**
   * Returns the minimal range that {@linkplain Range#contains(Comparable) contains} all of the
   * given values. The returned range is {@linkplain BoundType#CLOSED closed} on both ends.
   *
   * @throws ClassCastException if the values are not mutually comparable
   * @throws NoSuchElementException if {@code values} is empty
   * @throws NullPointerException if any of {@code values} is null
   * @since 14.0
   */
  public static <C extends Comparable<?>> Range<C> encloseAll(Iterable<C> values) {
    checkNotNull(values);
    if (values instanceof SortedSet) {
      SortedSet<C> set = (SortedSet<C>) values;
      Comparator<?> comparator = set.comparator();
      if (Ordering.<C>natural().equals(comparator) || comparator == null) {
        return closed(set.first(), set.last());
      }
    }
    Iterator<C> valueIterator = values.iterator();
    C min = checkNotNull(valueIterator.next());
