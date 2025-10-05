// Source-based slice around line 70
// Method: <com.google.common.collect.EnumBiMap: EnumBiMap create(Class,Class)>

  transient Class<V> valueTypeOrObjectUnderJ2cl;

  /**
   * Returns a new, empty {@code EnumBiMap} using the specified key and value types.
   *
   * @param keyType the key type
   * @param valueType the value type
   */
  public static <K extends Enum<K>, V extends Enum<V>> EnumBiMap<K, V> create(
      Class<K> keyType, Class<V> valueType) {
    return new EnumBiMap<>(keyType, valueType);
  }

  /**
   * Returns a new bimap with the same mappings as the specified map. If the specified map is an
   * {@code EnumBiMap}, the new bimap has the same types as the provided map. Otherwise, the
   * specified map must contain at least one mapping, in order to determine the key and value types.
   *
   * @param map the map whose mappings are to be placed in this map
   * @throws IllegalArgumentException if map is not an {@code EnumBiMap} instance and contains no
