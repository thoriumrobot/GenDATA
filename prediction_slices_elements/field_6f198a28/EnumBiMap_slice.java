// Source-based slice around line 60
// Method: com.google.common.collect.EnumBiMap.keyTypeOrObjectUnderJ2cl

   *
   * Then we declare the getters for these fields as @GwtIncompatible so that no one can try to use
   * them under J2CL—or, as an unfortunate side effect, under GWT. We do still give the fields
   * themselves their proper values under GWT, since GWT's EnumMap does need the Class instance.
   *
   * Note that sometimes these fields *do* have correct values under J2CL: They will if the caller
   * calls `create(Foo.class)`, rather than `create(map)`. That's fine; we just shouldn't rely on
   * it.
   */
  transient Class<K> keyTypeOrObjectUnderJ2cl;
  transient Class<V> valueTypeOrObjectUnderJ2cl;

  /**
   * Returns a new, empty {@code EnumBiMap} using the specified key and value types.
   *
   * @param keyType the key type
   * @param valueType the value type
   */
  public static <K extends Enum<K>, V extends Enum<V>> EnumBiMap<K, V> create(
      Class<K> keyType, Class<V> valueType) {
