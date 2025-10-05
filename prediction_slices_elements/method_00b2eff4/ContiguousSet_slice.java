// Source-based slice around line 167
// Method: <com.google.common.collect.ContiguousSet: ContiguousSet subSet(C,C)>

   * @since 12.0
   */
  @GwtIncompatible // NavigableSet
  @Override
  public ContiguousSet<C> headSet(C toElement, boolean inclusive) {
    return headSetImpl(checkNotNull(toElement), inclusive);
  }

  @Override
  public ContiguousSet<C> subSet(C fromElement, C toElement) {
    checkNotNull(fromElement);
    checkNotNull(toElement);
    checkArgument(comparator().compare(fromElement, toElement) <= 0);
    return subSetImpl(fromElement, true, toElement, false);
  }

  /**
   * @since 12.0
   */
  @GwtIncompatible // NavigableSet
