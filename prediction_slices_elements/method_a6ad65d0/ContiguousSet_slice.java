// Source-based slice around line 197
// Method: <com.google.common.collect.ContiguousSet: ContiguousSet tailSet(C,boolean)>

  public ContiguousSet<C> tailSet(C fromElement) {
    return tailSetImpl(checkNotNull(fromElement), true);
  }

  /**
   * @since 12.0
   */
  @GwtIncompatible // NavigableSet
  @Override
  public ContiguousSet<C> tailSet(C fromElement, boolean inclusive) {
    return tailSetImpl(checkNotNull(fromElement), inclusive);
  }

  /*
   * These methods perform most headSet, subSet, and tailSet logic, besides parameter validation.
   */
  @SuppressWarnings("MissingOverride") // Supermethod does not exist under GWT.
  abstract ContiguousSet<C> headSetImpl(C toElement, boolean inclusive);

  @SuppressWarnings("MissingOverride") // Supermethod does not exist under GWT.
