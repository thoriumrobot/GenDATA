// Source-based slice around line 205
// Method: <com.google.common.collect.ContiguousSet: ContiguousSet headSetImpl(C,boolean)>

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
  abstract ContiguousSet<C> subSetImpl(
      C fromElement, boolean fromInclusive, C toElement, boolean toInclusive);

  @SuppressWarnings("MissingOverride") // Supermethod does not exist under GWT.
  abstract ContiguousSet<C> tailSetImpl(C fromElement, boolean inclusive);

  /**
   * Returns the set of values that are contained in both this set and the other.
