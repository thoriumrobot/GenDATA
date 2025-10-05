// Source-based slice around line 287
// Method: <com.google.common.cache.CacheBuilderSpec: Long durationInNanos(long,TimeUnit)>

        && Objects.equals(
            durationInNanos(refreshDuration, refreshTimeUnit),
            durationInNanos(that.refreshDuration, that.refreshTimeUnit));
  }

  /**
   * Converts an expiration duration/unit pair into a single Long for hashing and equality. Uses
   * nanos to match CacheBuilder implementation.
   */
  private static @Nullable Long durationInNanos(long duration, @Nullable TimeUnit unit) {
    return (unit == null) ? null : unit.toNanos(duration);
  }

  /** Base class for parsing integers. */
  abstract static class IntegerParser implements ValueParser {
    protected abstract void parseInteger(CacheBuilderSpec spec, int value);

    @Override
    public void parse(CacheBuilderSpec spec, String key, @Nullable String value) {
      if (isNullOrEmpty(value)) {
