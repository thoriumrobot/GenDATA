// Source-based slice around line 134
// Method: <com.google.common.collect.EnumBiMap: V checkValue(V)>

    return valueTypeOrObjectUnderJ2cl;
  }

  @Override
  K checkKey(K key) {
    return checkNotNull(key);
  }

  @Override
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
