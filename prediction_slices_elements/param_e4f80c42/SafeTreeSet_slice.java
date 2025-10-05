// Source-based slice around line 208
// Method: <com.google.common.collect.testing.SafeTreeSet: SortedSet subSet(E,E)>

  @Override
  public NavigableSet<E> subSet(
      E fromElement, boolean fromInclusive, E toElement, boolean toInclusive) {
    return new SafeTreeSet<>(
        delegate.subSet(
            checkValid(fromElement), fromInclusive, checkValid(toElement), toInclusive));
  }

  @Override
  public SortedSet<E> subSet(E fromElement, E toElement) {
    return subSet(fromElement, true, toElement, false);
  }

  @Override
  public SortedSet<E> tailSet(E fromElement) {
    return tailSet(fromElement, true);
  }

  @Override
  public NavigableSet<E> tailSet(E fromElement, boolean inclusive) {
