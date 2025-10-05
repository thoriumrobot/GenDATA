// Source-based slice around line 188
// Method: <com.google.common.collect.ContiguousSet: ContiguousSet tailSet(C)>

  public ContiguousSet<C> subSet(
      C fromElement, boolean fromInclusive, C toElement, boolean toInclusive) {
    checkNotNull(fromElement);
    checkNotNull(toElement);
    checkArgument(comparator().compare(fromElement, toElement) <= 0);
    return subSetImpl(fromElement, fromInclusive, toElement, toInclusive);
  }

  @Override
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
