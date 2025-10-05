// Source-based slice around line 125
// Method: <com.google.common.cache.LoadingCache: ImmutableMap getAll(Iterable)>

   *     ExecutionException} is thrown <a
   *     href="https://github.com/google/guava/wiki/CachesExplained#interruption">even if
   *     computation was interrupted by an {@code InterruptedException}</a>.)
   * @throws UncheckedExecutionException if an unchecked exception was thrown while loading the
   *     values
   * @throws ExecutionError if an error was thrown while loading the values
   * @since 11.0
   */
  @CanIgnoreReturnValue // TODO(b/27479612): consider removing this
  ImmutableMap<K, V> getAll(Iterable<? extends K> keys) throws ExecutionException;

  /**
   * @deprecated Provided to satisfy the {@code Function} interface; use {@link #get} or {@link
   *     #getUnchecked} instead.
   * @throws UncheckedExecutionException if an exception was thrown while loading the value. (As
   *     described in the documentation for {@link #getUnchecked}, {@code LoadingCache} should be
   *     used as a {@code Function} only with cache loaders that throw only unchecked exceptions.)
   */
  @Deprecated
  @Override
