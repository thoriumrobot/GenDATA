// Source-based slice around line 2791
// Method: <com.google.common.collect.MapMakerInternalMap: Object writeReplace()>

    public void clear() {
      MapMakerInternalMap.this.clear();
    }
  }

  // Serialization Support

  private static final long serialVersionUID = 5;

  Object writeReplace() {
    return new SerializationProxy<>(
        entryHelper.keyStrength(),
        entryHelper.valueStrength(),
        keyEquivalence,
        entryHelper.valueStrength().defaultEquivalence(),
        concurrencyLevel,
        this);
  }

  @J2ktIncompatible // java.io.ObjectInputStream
