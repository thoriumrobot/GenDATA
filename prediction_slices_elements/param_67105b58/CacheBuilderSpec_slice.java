// Source-based slice around line 257
// Method: <com.google.common.cache.CacheBuilderSpec: boolean equals(Object)>

        keyStrength,
        valueStrength,
        recordStats,
        durationInNanos(writeExpirationDuration, writeExpirationTimeUnit),
        durationInNanos(accessExpirationDuration, accessExpirationTimeUnit),
        durationInNanos(refreshDuration, refreshTimeUnit));
  }

  @Override
  public boolean equals(@Nullable Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof CacheBuilderSpec)) {
      return false;
    }
    CacheBuilderSpec that = (CacheBuilderSpec) obj;
    return Objects.equals(initialCapacity, that.initialCapacity)
        && Objects.equals(maximumSize, that.maximumSize)
        && Objects.equals(maximumWeight, that.maximumWeight)
