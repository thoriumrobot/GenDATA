// Source-based slice around line 405
// Method: <com.google.common.collect.ImmutableMultimap: ImmutableMultimap copyOf(Iterable)>


  /**
   * Returns an immutable multimap containing the specified entries. The returned multimap iterates
   * over keys in the order they were first encountered in the input, and the values for each key
   * are iterated in the order they were encountered.
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
