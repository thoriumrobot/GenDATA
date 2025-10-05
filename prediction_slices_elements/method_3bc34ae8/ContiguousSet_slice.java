// Source-based slice around line 251
// Method: <com.google.common.collect.ContiguousSet: String toString()>


  @Override
  @GwtIncompatible // NavigableSet
  ImmutableSortedSet<C> createDescendingSet() {
    return new DescendingImmutableSortedSet<>(this);
  }

  /** Returns a shorthand representation of the contents such as {@code "[1..100]"}. */
  @Override
  public String toString() {
    return range().toString();
  }

  /**
   * Not supported. {@code ContiguousSet} instances are constructed with {@link #create}. This
   * method exists only to hide {@link ImmutableSet#builder} from consumers of {@code
   * ContiguousSet}.
   *
   * @throws UnsupportedOperationException always
   * @deprecated Use {@link #create}.
