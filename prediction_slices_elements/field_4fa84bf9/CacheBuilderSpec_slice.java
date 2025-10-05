// Source-based slice around line 123
// Method: com.google.common.cache.CacheBuilderSpec.writeExpirationTimeUnit


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

  /** Specification; used for toParseableString(). */
  private final String specification;

  private CacheBuilderSpec(String specification) {
    this.specification = specification;
