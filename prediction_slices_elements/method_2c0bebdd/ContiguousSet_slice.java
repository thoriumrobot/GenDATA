// Source-based slice around line 220
// Method: <com.google.common.collect.ContiguousSet: ContiguousSet intersection(ContiguousSet)>

  @SuppressWarnings("MissingOverride") // Supermethod does not exist under GWT.
  abstract ContiguousSet<C> tailSetImpl(C fromElement, boolean inclusive);

  /**
   * Returns the set of values that are contained in both this set and the other.
   *
   * <p>This method should always be used instead of {@link Sets#intersection} for {@link
   * ContiguousSet} instances.
   */
  public abstract ContiguousSet<C> intersection(ContiguousSet<C> other);

  /**
   * Returns a range, closed on both ends, whose endpoints are the minimum and maximum values
   * contained in this set. This is equivalent to {@code range(CLOSED, CLOSED)}.
   *
   * @throws NoSuchElementException if this set is empty
   */
  public abstract Range<C> range();

  /**
