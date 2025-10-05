// Source-based slice around line 53
// Method: <com.google.common.collect.ImmutableSortedAsList: int indexOf(Object)>

  public Comparator<? super E> comparator() {
    return delegateCollection().comparator();
  }

  // Override indexOf() and lastIndexOf() to be O(log N) instead of O(N).

  @GwtIncompatible // ImmutableSortedSet.indexOf
  // TODO(cpovirk): consider manual binary search under GWT to preserve O(log N) lookup
  @Override
  public int indexOf(@Nullable Object target) {
    int index = delegateCollection().indexOf(target);

    // TODO(kevinb): reconsider if it's really worth making feeble attempts at
    // sanity for inconsistent comparators.

    // The equals() check is needed when the comparator isn't compatible with
    // equals().
    return (index >= 0 && get(index).equals(target)) ? index : -1;
  }

