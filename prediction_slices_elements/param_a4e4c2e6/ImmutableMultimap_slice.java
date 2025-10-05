// Source-based slice around line 89
// Method: <com.google.common.collect.ImmutableMultimap: ImmutableMultimap of(K,V)>

   * Returns an empty multimap.
   *
   * <p><b>Performance note:</b> the instance returned is a singleton.
   */
  public static <K, V> ImmutableMultimap<K, V> of() {
    return ImmutableListMultimap.of();
  }

  /** Returns an immutable multimap containing a single entry. */
  public static <K, V> ImmutableMultimap<K, V> of(K k1, V v1) {
    return ImmutableListMultimap.of(k1, v1);
  }

  /** Returns an immutable multimap containing the given entries, in order. */
  public static <K, V> ImmutableMultimap<K, V> of(K k1, V v1, K k2, V v2) {
    return ImmutableListMultimap.of(k1, v1, k2, v2);
  }

  /**
   * Returns an immutable multimap containing the given entries, in the "key-grouped" insertion
