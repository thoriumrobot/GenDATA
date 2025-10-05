// Source-based slice around line 126
// Method: <com.google.common.cache.Cache: void put(K,V)>

  /**
   * Associates {@code value} with {@code key} in this cache. If the cache previously contained a
   * value associated with {@code key}, the old value is replaced by {@code value}.
   *
   * <p>Prefer {@link #get(Object, Callable)} when using the conventional "if cached, return;
   * otherwise create, cache and return" pattern.
   *
   * @since 11.0
   */
  void put(K key, V value);

  /**
   * Copies all of the mappings from the specified map to the cache. The effect of this call is
   * equivalent to that of calling {@code put(k, v)} on this map once for each mapping from key
   * {@code k} to value {@code v} in the specified map. The behavior of this operation is undefined
   * if the specified map is modified while the operation is in progress.
   *
   * @since 12.0
   */
  void putAll(Map<? extends K, ? extends V> m);
