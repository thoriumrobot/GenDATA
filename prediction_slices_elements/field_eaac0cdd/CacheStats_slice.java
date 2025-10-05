// Source-based slice around line 64
// Method: com.google.common.cache.CacheStats.loadSuccessCount

 * Callable)}, or {@link LoadingCache#getAll(Iterable)}.
 *
 * @author Charles Fry
 * @since 10.0
 */
@GwtCompatible
public final class CacheStats {
  private final long hitCount;
  private final long missCount;
  private final long loadSuccessCount;
  private final long loadExceptionCount;

  @SuppressWarnings("GoodTime") // should be a java.time.Duration
  private final long totalLoadTime;

  private final long evictionCount;

  /**
   * Constructs a new {@code CacheStats} instance.
   *
