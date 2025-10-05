// Source-based slice around line 201
// Method: <com.google.common.collect.testing.SafeTreeSet: NavigableSet subSet(E,boolean,E,boolean)>

  }

  @Override
  public int size() {
    return delegate.size();
  }

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

