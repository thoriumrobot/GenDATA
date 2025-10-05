// Source-based slice around line 237
// Method: <com.google.common.cache.CacheBuilderSpec: String toString()>

  public String toParsableString() {
    return specification;
  }

  /**
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
