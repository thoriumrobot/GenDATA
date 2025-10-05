// Source-based slice around line 71
// Method: <com.google.common.collect.ImmutableSortedAsList: boolean contains(Object)>

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
  /*
   * TODO(cpovirk): if we start to override indexOf/lastIndexOf under GWT, we'll want some way to
   * override subList to return an ImmutableSortedAsList for better performance. Right now, I'm not
   * sure there's any performance hit from our failure to override subListUnchecked under GWT
   */
