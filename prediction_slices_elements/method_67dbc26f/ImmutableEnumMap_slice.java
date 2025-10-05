// Source-based slice around line 120
// Method: <com.google.common.collect.ImmutableEnumMap: Object writeReplace()>


  @Override
  boolean isPartialView() {
    return false;
  }

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
