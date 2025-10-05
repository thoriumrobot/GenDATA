// Source-based slice around line 127
// Method: com.google.common.cache.CacheBuilderSpec.refreshTimeUnit

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
   * Creates a CacheBuilderSpec from a string.
