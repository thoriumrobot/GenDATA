// Source-based slice around line 118
// Method: com.google.common.cache.CacheBuilderSpec.concurrencyLevel

          .put("expireAfterAccess", new AccessDurationParser())
          .put("expireAfterWrite", new WriteDurationParser())
          .put("refreshAfterWrite", new RefreshDurationParser())
          .put("refreshInterval", new RefreshDurationParser())
          .buildOrThrow();

  @VisibleForTesting @Nullable Integer initialCapacity;
  @VisibleForTesting @Nullable Long maximumSize;
  @VisibleForTesting @Nullable Long maximumWeight;
  @VisibleForTesting @Nullable Integer concurrencyLevel;
  @VisibleForTesting @Nullable Strength keyStrength;
  @VisibleForTesting @Nullable Strength valueStrength;
  @VisibleForTesting @Nullable Boolean recordStats;
  @VisibleForTesting long writeExpirationDuration;
  @VisibleForTesting @Nullable TimeUnit writeExpirationTimeUnit;
  @VisibleForTesting long accessExpirationDuration;
  @VisibleForTesting @Nullable TimeUnit accessExpirationTimeUnit;
  @VisibleForTesting long refreshDuration;
  @VisibleForTesting @Nullable TimeUnit refreshTimeUnit;

