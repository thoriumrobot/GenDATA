// Source-based slice around line 213
// Method: <com.google.common.collect.testing.SafeTreeSet: SortedSet tailSet(E)>

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
    return new SafeTreeSet<>(delegate.tailSet(checkValid(fromElement), inclusive));
  }

  @Override
  public Object[] toArray() {
