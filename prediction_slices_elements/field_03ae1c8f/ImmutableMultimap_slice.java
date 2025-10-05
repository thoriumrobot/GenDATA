// Source-based slice around line 410
// Method: com.google.common.collect.ImmutableMultimap.map

   *
   * @throws NullPointerException if any key, value, or entry is null
   * @since 19.0
   */
  public static <K, V> ImmutableMultimap<K, V> copyOf(
      Iterable<? extends Entry<? extends K, ? extends V>> entries) {
    return ImmutableListMultimap.copyOf(entries);
  }

  final transient ImmutableMap<K, ? extends ImmutableCollection<V>> map;
  final transient int size;

  // These constants allow the deserialization code to set final fields. This
  // holder class makes sure they are not initialized unless an instance is
  // deserialized.
  @GwtIncompatible
  @J2ktIncompatible
  static final class FieldSettersHolder {
    static final Serialization.FieldSetter<? super ImmutableMultimap<?, ?>> MAP_FIELD_SETTER =
        Serialization.getFieldSetter(ImmutableMultimap.class, "map");
