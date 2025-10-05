// Source-based slice around line 138
// Method: <com.google.common.collect.ImmutableMapEntrySet: void readObject(ObjectInputStream)>

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
  @J2ktIncompatible
  private static final class EntrySetSerializedForm<K, V> implements Serializable {
    final ImmutableMap<K, V> map;

    EntrySetSerializedForm(ImmutableMap<K, V> map) {
      this.map = map;
