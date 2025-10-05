// Source-based slice around line 124
// Method: <com.google.common.collect.EnumBiMap: Class valueType()>


  /** Returns the associated key type. */
  @GwtIncompatible
  public Class<K> keyType() {
    return keyTypeOrObjectUnderJ2cl;
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
