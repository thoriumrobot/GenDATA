// Source-based slice around line 66
// Method: <com.google.common.collect.ImmutableSortedAsList: int lastIndexOf(Object)>

    // sanity for inconsistent comparators.

    // The equals() check is needed when the comparator isn't compatible with
    // equals().
    return (index >= 0 && get(index).equals(target)) ? index : -1;
  }

  @GwtIncompatible // ImmutableSortedSet.indexOf
  @Override
  public int lastIndexOf(@Nullable Object target) {
    return indexOf(target);
  }

  @Override
  public boolean contains(@Nullable Object target) {
    // Necessary for ISS's with comparators inconsistent with equals.
    return indexOf(target) >= 0;
  }

  @GwtIncompatible // super.subListUnchecked does not exist; inherited subList is valid if slow
