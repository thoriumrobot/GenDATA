// Source-based slice around line 141
// Method: <com.google.common.collect.ImmutableEnumSet: Object writeReplace()>

  }

  @Override
  public String toString() {
    return delegate.toString();
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
