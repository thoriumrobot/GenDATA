// Source-based slice around line 212
// Method: <com.google.common.collect.ContiguousSet: ContiguousSet tailSetImpl(C,boolean)>

   */
  @SuppressWarnings("MissingOverride") // Supermethod does not exist under GWT.
  abstract ContiguousSet<C> headSetImpl(C toElement, boolean inclusive);

  @SuppressWarnings("MissingOverride") // Supermethod does not exist under GWT.
  abstract ContiguousSet<C> subSetImpl(
      C fromElement, boolean fromInclusive, C toElement, boolean toInclusive);

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
