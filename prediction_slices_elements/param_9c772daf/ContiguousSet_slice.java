// Source-based slice around line 180
// Method: <com.google.common.collect.ContiguousSet: ContiguousSet subSet(C,boolean,C,boolean)>

    return subSetImpl(fromElement, true, toElement, false);
  }

  /**
   * @since 12.0
   */
  @GwtIncompatible // NavigableSet
  @Override
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
