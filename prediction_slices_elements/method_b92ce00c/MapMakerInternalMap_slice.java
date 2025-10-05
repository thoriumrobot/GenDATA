// Source-based slice around line 2802
// Method: <com.google.common.collect.MapMakerInternalMap: void readObject(ObjectInputStream)>

        entryHelper.keyStrength(),
        entryHelper.valueStrength(),
        keyEquivalence,
        entryHelper.valueStrength().defaultEquivalence(),
        concurrencyLevel,
        this);
  }

  @J2ktIncompatible // java.io.ObjectInputStream
  private void readObject(ObjectInputStream in) throws InvalidObjectException {
    throw new InvalidObjectException("Use SerializationProxy");
  }

  /**
   * The actual object that gets serialized. Unfortunately, readResolve() doesn't get called when a
   * circular dependency is present, so the proxy must be able to behave as the map itself.
   */
  abstract static class AbstractSerializationProxy<K, V> extends ForwardingConcurrentMap<K, V>
      implements Serializable {
    private static final long serialVersionUID = 3;
