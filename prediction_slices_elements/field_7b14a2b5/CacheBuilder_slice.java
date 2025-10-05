// Source-based slice around line 291
// Method: com.google.common.cache.CacheBuilder.keyStrength


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

  @SuppressWarnings("GoodTime") // should be a Duration
  long expireAfterAccessNanos = UNSET_INT;

  @SuppressWarnings("GoodTime") // should be a Duration
  long refreshNanos = UNSET_INT;
