// Source-based slice around line 286
// Method: com.google.common.cache.CacheBuilder.concurrencyLevel

  private static final class LoggerHolder {
    static final Logger logger = Logger.getLogger(CacheBuilder.class.getName());
  }

  static final int UNSET_INT = -1;

  boolean strictParsing = true;

  int initialCapacity = UNSET_INT;
  int concurrencyLevel = UNSET_INT;
  long maximumSize = UNSET_INT;
  long maximumWeight = UNSET_INT;
  @Nullable Weigher<? super K, ? super V> weigher;

  @Nullable Strength keyStrength;
  @Nullable Strength valueStrength;

  @SuppressWarnings("GoodTime") // should be a Duration
  long expireAfterWriteNanos = UNSET_INT;

