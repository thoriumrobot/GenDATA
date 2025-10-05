// Source-based slice around line 71
// Method: <com.google.common.cache.LoadingCache: V get(K)>

   * @throws ExecutionException if a checked exception was thrown while loading the value. ({@code
   *     ExecutionException} is thrown <a
   *     href="https://github.com/google/guava/wiki/CachesExplained#interruption">even if
   *     computation was interrupted by an {@code InterruptedException}</a>.)
   * @throws UncheckedExecutionException if an unchecked exception was thrown while loading the
   *     value
   * @throws ExecutionError if an error was thrown while loading the value
   */
  @CanIgnoreReturnValue // TODO(b/27479612): consider removing this?
  V get(K key) throws ExecutionException;

  /**
   * Returns the value associated with {@code key} in this cache, first loading that value if
   * necessary. No observable state associated with this cache is modified until loading completes.
   * Unlike {@link #get}, this method does not throw a checked exception, and thus should only be
   * used in situations where checked exceptions are not thrown by the cache loader.
   *
   * <p>If another call to {@link #get} or {@link #getUnchecked} is currently loading the value for
   * {@code key}, simply waits for that thread to finish and returns its loaded value. Note that
   * multiple threads can concurrently load values for distinct keys.
