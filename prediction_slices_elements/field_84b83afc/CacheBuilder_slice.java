// Source-based slice around line 200
// Method: com.google.common.cache.CacheBuilder.DEFAULT_EXPIRATION_NANOS

 * @author Kevin Bourrillion
 * @since 10.0
 */
@GwtCompatible
public final class CacheBuilder<K, V> {
  private static final int DEFAULT_INITIAL_CAPACITY = 16;
  private static final int DEFAULT_CONCURRENCY_LEVEL = 4;

  @SuppressWarnings("GoodTime") // should be a Duration
  private static final int DEFAULT_EXPIRATION_NANOS = 0;

  @SuppressWarnings("GoodTime") // should be a Duration
  private static final int DEFAULT_REFRESH_NANOS = 0;

  static final Supplier<? extends StatsCounter> NULL_STATS_COUNTER =
      Suppliers.ofInstance(
          new StatsCounter() {
            @Override
            public void recordHits(int count) {}

