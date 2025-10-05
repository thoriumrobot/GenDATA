// Source-based slice around line 143
// Method: <com.google.common.collect.EnumBiMap: void writeObject(ObjectOutputStream)>

  V checkValue(V value) {
    return checkNotNull(value);
  }

  /**
   * @serialData the key class, value class, number of entries, first key, first value, second key,
   *     second value, and so on.
   */
  @GwtIncompatible // java.io.ObjectOutputStream
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
