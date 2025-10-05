// Source-based slice around line 110
// Method: <com.google.common.collect.ImmutableMultimap: ImmutableMultimap of(K,V,K,V,K,V,K,V)>

   */
  public static <K, V> ImmutableMultimap<K, V> of(K k1, V v1, K k2, V v2, K k3, V v3) {
    return ImmutableListMultimap.of(k1, v1, k2, v2, k3, v3);
  }

  /**
   * Returns an immutable multimap containing the given entries, in the "key-grouped" insertion
   * order described in the <a href="#iteration">class documentation</a>.
   */
  public static <K, V> ImmutableMultimap<K, V> of(K k1, V v1, K k2, V v2, K k3, V v3, K k4, V v4) {
    return ImmutableListMultimap.of(k1, v1, k2, v2, k3, v3, k4, v4);
  }

  /**
   * Returns an immutable multimap containing the given entries, in the "key-grouped" insertion
   * order described in the <a href="#iteration">class documentation</a>.
   */
  public static <K, V> ImmutableMultimap<K, V> of(
      K k1, V v1, K k2, V v2, K k3, V v3, K k4, V v4, K k5, V v5) {
    return ImmutableListMultimap.of(k1, v1, k2, v2, k3, v3, k4, v4, k5, v5);
