// Source-based slice around line 152
// Method: <com.google.common.collect.EnumBiMap: void readObject(ObjectInputStream)>

  private void writeObject(ObjectOutputStream stream) throws IOException {
    stream.defaultWriteObject();
    stream.writeObject(keyTypeOrObjectUnderJ2cl);
    stream.writeObject(valueTypeOrObjectUnderJ2cl);
    Serialization.writeMap(this, stream);
  }

  @SuppressWarnings("unchecked") // reading fields populated by writeObject
  @GwtIncompatible // java.io.ObjectInputStream
  private void readObject(ObjectInputStream stream) throws IOException, ClassNotFoundException {
    stream.defaultReadObject();
    keyTypeOrObjectUnderJ2cl = (Class<K>) requireNonNull(stream.readObject());
    valueTypeOrObjectUnderJ2cl = (Class<V>) requireNonNull(stream.readObject());
    setDelegates(
        new EnumMap<>(keyTypeOrObjectUnderJ2cl), new EnumMap<>(valueTypeOrObjectUnderJ2cl));
    Serialization.populateMap(this, stream);
  }

  @GwtIncompatible @J2ktIncompatible private static final long serialVersionUID = 0;
}
