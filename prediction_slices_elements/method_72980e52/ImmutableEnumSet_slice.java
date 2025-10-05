// Source-based slice around line 146
// Method: <com.google.common.collect.ImmutableEnumSet: void readObject(ObjectInputStream)>

  }

  @Override
  @J2ktIncompatible // serialization
  Object writeReplace() {
    return new EnumSerializedForm<E>(delegate);
  }

  @J2ktIncompatible // serialization
  private void readObject(ObjectInputStream stream) throws InvalidObjectException {
    throw new InvalidObjectException("Use SerializedForm");
  }

  /*
   * This class is used to serialize ImmutableEnumSet instances.
   */
  @J2ktIncompatible // serialization
  private static final class EnumSerializedForm<E extends Enum<E>> implements Serializable {
    final EnumSet<E> delegate;

