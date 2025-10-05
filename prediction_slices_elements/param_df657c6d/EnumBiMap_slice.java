// Source-based slice around line 129
// Method: <com.google.common.collect.EnumBiMap: K checkKey(K)>

  }

  /** Returns the associated value type. */
  @GwtIncompatible
  public Class<V> valueType() {
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
