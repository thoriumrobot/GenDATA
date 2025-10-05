// Source-based slice around line 2789
// Method: com.google.common.collect.MapMakerInternalMap.serialVersionUID


    @Override
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
