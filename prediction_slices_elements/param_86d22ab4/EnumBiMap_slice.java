// Source-based slice around line 108
// Method: <com.google.common.collect.EnumBiMap: Class inferValueTypeOrObjectUnderJ2cl(Map)>

      return ((EnumBiMap<K, ?>) map).keyTypeOrObjectUnderJ2cl;
    }
    if (map instanceof EnumHashBiMap) {
      return ((EnumHashBiMap<K, ?>) map).keyTypeOrObjectUnderJ2cl;
    }
    checkArgument(!map.isEmpty());
    return getDeclaringClassOrObjectForJ2cl(map.keySet().iterator().next());
  }

  private static <V extends Enum<V>> Class<V> inferValueTypeOrObjectUnderJ2cl(Map<?, V> map) {
    if (map instanceof EnumBiMap) {
      return ((EnumBiMap<?, V>) map).valueTypeOrObjectUnderJ2cl;
    }
    checkArgument(!map.isEmpty());
    return getDeclaringClassOrObjectForJ2cl(map.values().iterator().next());
  }

  /** Returns the associated key type. */
  @GwtIncompatible
  public Class<K> keyType() {
