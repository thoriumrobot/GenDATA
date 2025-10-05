// Source-based slice around line 126
// Method: com.google.common.cache.CacheBuilderSpec.refreshDuration

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

  /** Specification; used for toParseableString(). */
  private final String specification;

  private CacheBuilderSpec(String specification) {
    this.specification = specification;
  }

  /**
