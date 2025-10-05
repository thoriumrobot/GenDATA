// Source-based slice around line 94
// Method: <com.google.common.collect.ImmutableMultimap: ImmutableMultimap of(K,V,K,V)>

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
   * order described in the <a href="#iteration">class documentation</a>.
   */
  public static <K, V> ImmutableMultimap<K, V> of(K k1, V v1, K k2, V v2, K k3, V v3) {
    return ImmutableListMultimap.of(k1, v1, k2, v2, k3, v3);
  }
