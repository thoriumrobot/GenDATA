// Source-based slice around line 118
// Method: <com.google.common.collect.EnumBiMap: Class keyType()>

    if (map instanceof EnumBiMap) {
      return ((EnumBiMap<?, V>) map).valueTypeOrObjectUnderJ2cl;
    }
    checkArgument(!map.isEmpty());
    return getDeclaringClassOrObjectForJ2cl(map.values().iterator().next());
  }

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
