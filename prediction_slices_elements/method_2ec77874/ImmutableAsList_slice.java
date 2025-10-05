// Source-based slice around line 87
// Method: <com.google.common.collect.ImmutableAsList: Object writeReplace()>

  @GwtIncompatible
  @J2ktIncompatible
    private void readObject(ObjectInputStream stream) throws InvalidObjectException {
    throw new InvalidObjectException("Use SerializedForm");
  }

  @GwtIncompatible
  @J2ktIncompatible
    @Override
  Object writeReplace() {
    return new SerializedForm(delegateCollection());
  }
}
