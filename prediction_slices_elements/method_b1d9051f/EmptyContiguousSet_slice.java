// Source-based slice around line 179
// Method: <com.google.common.collect.EmptyContiguousSet: ImmutableSortedSet createDescendingSet()>


  @GwtIncompatible
  @J2ktIncompatible
    private void readObject(ObjectInputStream stream) throws InvalidObjectException {
    throw new InvalidObjectException("Use SerializedForm");
  }

  @GwtIncompatible // NavigableSet
  @Override
  ImmutableSortedSet<C> createDescendingSet() {
    return ImmutableSortedSet.emptySet(Ordering.<C>natural().reverse());
  }
}
