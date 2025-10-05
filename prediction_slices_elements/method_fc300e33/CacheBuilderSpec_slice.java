// Source-based slice around line 242
// Method: <com.google.common.cache.CacheBuilderSpec: int hashCode()>

   * Returns a string representation for this CacheBuilderSpec instance. The form of this
   * representation is not guaranteed.
   */
  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this).addValue(toParsableString()).toString();
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        initialCapacity,
        maximumSize,
        maximumWeight,
        concurrencyLevel,
        keyStrength,
        valueStrength,
        recordStats,
        durationInNanos(writeExpirationDuration, writeExpirationTimeUnit),
        durationInNanos(accessExpirationDuration, accessExpirationTimeUnit),
