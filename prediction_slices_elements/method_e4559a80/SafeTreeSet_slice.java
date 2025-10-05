// Source-based slice around line 195
// Method: <com.google.common.collect.testing.SafeTreeSet: int size()>

    return delegate.removeAll(c);
  }

  @Override
  public boolean retainAll(Collection<?> c) {
    return delegate.retainAll(c);
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
