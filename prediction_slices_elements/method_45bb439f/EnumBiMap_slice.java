// Source-based slice around line 97
// Method: <com.google.common.collect.EnumBiMap: Class inferKeyTypeOrObjectUnderJ2cl(Map)>

  }

  private EnumBiMap(Class<K> keyTypeOrObjectUnderJ2cl, Class<V> valueTypeOrObjectUnderJ2cl) {
    super(
        new EnumMap<K, V>(keyTypeOrObjectUnderJ2cl), new EnumMap<V, K>(valueTypeOrObjectUnderJ2cl));
    this.keyTypeOrObjectUnderJ2cl = keyTypeOrObjectUnderJ2cl;
    this.valueTypeOrObjectUnderJ2cl = valueTypeOrObjectUnderJ2cl;
  }

  static <K extends Enum<K>> Class<K> inferKeyTypeOrObjectUnderJ2cl(Map<K, ?> map) {
    if (map instanceof EnumBiMap) {
      return ((EnumBiMap<K, ?>) map).keyTypeOrObjectUnderJ2cl;
    }
    if (map instanceof EnumHashBiMap) {
      return ((EnumHashBiMap<K, ?>) map).keyTypeOrObjectUnderJ2cl;
    }
    checkArgument(!map.isEmpty());
    return getDeclaringClassOrObjectForJ2cl(map.keySet().iterator().next());
  }

