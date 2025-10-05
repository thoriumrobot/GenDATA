// Source-based slice around line 132
// Method: <com.google.common.collect.ImmutableMapEntrySet: Object writeReplace()>


  @Override
  public int hashCode() {
    return map().hashCode();
  }

  @GwtIncompatible
  @J2ktIncompatible
    @Override
  Object writeReplace() {
    return new EntrySetSerializedForm<>(map());
  }

  @GwtIncompatible
  @J2ktIncompatible
    private void readObject(ObjectInputStream stream) throws InvalidObjectException {
    throw new InvalidObjectException("Use EntrySetSerializedForm");
  }

  @GwtIncompatible
