// Source-based slice around line 125
// Method: <com.google.common.collect.ImmutableEnumMap: void readObject(ObjectInputStream)>


  // All callers of the constructor are restricted to <K extends Enum<K>>.
  @Override
  @J2ktIncompatible // serialization
  Object writeReplace() {
    return new EnumSerializedForm<>(delegate);
  }

  @J2ktIncompatible // serialization
  private void readObject(ObjectInputStream stream) throws InvalidObjectException {
    throw new InvalidObjectException("Use EnumSerializedForm");
  }

  /*
   * This class is used to serialize ImmutableEnumMap instances.
   */
  @J2ktIncompatible // serialization
  private static final class EnumSerializedForm<K extends Enum<K>, V> implements Serializable {
    final EnumMap<K, V> delegate;

