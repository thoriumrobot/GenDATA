// Source-based slice around line 65
// Method: com.google.common.cache.CacheStats.loadExceptionCount

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
   * <p>Five parameters of the same type in a row is a bad thing, but this class is not constructed
