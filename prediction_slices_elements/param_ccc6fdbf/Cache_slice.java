// Source-based slice around line 103
// Method: <com.google.common.cache.Cache: V get(K,Callable)>

   * <p>No observable state associated with this cache is modified until loading completes.
   *
   * @throws ExecutionException if a checked exception was thrown while loading the value
   * @throws UncheckedExecutionException if an unchecked exception was thrown while loading the
   *     value
   * @throws ExecutionError if an error was thrown while loading the value
   * @since 11.0
   */
  @CanIgnoreReturnValue // TODO(b/27479612): consider removing this
  V get(K key, Callable<? extends V> loader) throws ExecutionException;

  /**
   * Returns a map of the values associated with {@code keys} in this cache. The returned map will
   * only contain entries which are already present in the cache.
   *
   * @since 11.0
   */
  /*
   * <? extends Object> is mostly the same as <?> to plain Java. But to nullness checkers, they
   * differ: <? extends Object> means "non-null types," while <?> means "all types."
