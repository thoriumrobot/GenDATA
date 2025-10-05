// Source-based slice around line 267
// Method: <com.google.common.collect.RegularContiguousSet: void readObject(ObjectInputStream)>

  @GwtIncompatible
  @J2ktIncompatible
    @Override
  Object writeReplace() {
    return new SerializedForm<>(range, domain);
  }

  @GwtIncompatible
  @J2ktIncompatible
    private void readObject(ObjectInputStream stream) throws InvalidObjectException {
    throw new InvalidObjectException("Use SerializedForm");
  }

  @GwtIncompatible @J2ktIncompatible private static final long serialVersionUID = 0;
}
