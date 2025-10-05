// Source-based slice around line 115
// Method: <com.google.common.cache.Cache: ImmutableMap getAllPresent(Iterable)>

   * Returns a map of the values associated with {@code keys} in this cache. The returned map will
   * only contain entries which are already present in the cache.
   *
   * @since 11.0
   */
  /*
   * <? extends Object> is mostly the same as <?> to plain Java. But to nullness checkers, they
   * differ: <? extends Object> means "non-null types," while <?> means "all types."
   */
  ImmutableMap<K, V> getAllPresent(Iterable<? extends Object> keys);

  /**
   * Associates {@code value} with {@code key} in this cache. If the cache previously contained a
   * value associated with {@code key}, the old value is replaced by {@code value}.
   *
   * <p>Prefer {@link #get(Object, Callable)} when using the conventional "if cached, return;
   * otherwise create, cache and return" pattern.
   *
   * @since 11.0
   */
