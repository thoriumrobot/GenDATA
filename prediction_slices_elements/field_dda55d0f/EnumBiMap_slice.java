// Source-based slice around line 161
// Method: com.google.common.collect.EnumBiMap.serialVersionUID

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
