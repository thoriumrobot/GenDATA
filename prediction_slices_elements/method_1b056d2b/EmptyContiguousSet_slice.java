// Source-based slice around line 173
// Method: <com.google.common.collect.EmptyContiguousSet: void readObject(ObjectInputStream)>

  @GwtIncompatible
  @J2ktIncompatible
    @Override
  Object writeReplace() {
    return new SerializedForm<>(domain);
  }

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
