// Source-based slice around line 83
// Method: <com.google.common.collect.EnumBiMap: EnumBiMap create(Map)>

  /**
   * Returns a new bimap with the same mappings as the specified map. If the specified map is an
   * {@code EnumBiMap}, the new bimap has the same types as the provided map. Otherwise, the
   * specified map must contain at least one mapping, in order to determine the key and value types.
   *
   * @param map the map whose mappings are to be placed in this map
   * @throws IllegalArgumentException if map is not an {@code EnumBiMap} instance and contains no
   *     mappings
   */
  public static <K extends Enum<K>, V extends Enum<V>> EnumBiMap<K, V> create(Map<K, V> map) {
    EnumBiMap<K, V> bimap =
        create(inferKeyTypeOrObjectUnderJ2cl(map), inferValueTypeOrObjectUnderJ2cl(map));
    bimap.putAll(map);
    return bimap;
  }

  private EnumBiMap(Class<K> keyTypeOrObjectUnderJ2cl, Class<V> valueTypeOrObjectUnderJ2cl) {
    super(
        new EnumMap<K, V>(keyTypeOrObjectUnderJ2cl), new EnumMap<V, K>(valueTypeOrObjectUnderJ2cl));
    this.keyTypeOrObjectUnderJ2cl = keyTypeOrObjectUnderJ2cl;
