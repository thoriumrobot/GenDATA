// Source-based slice around line 162
// Method: <com.google.common.collect.ContiguousSet: ContiguousSet headSet(C,boolean)>

  public ContiguousSet<C> headSet(C toElement) {
    return headSetImpl(checkNotNull(toElement), false);
  }

  /**
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
