// Source-based slice around line 228
// Method: <com.google.common.cache.CacheBuilderSpec: String toParsableString()>


    return builder;
  }

  /**
   * Returns a string that can be used to parse an equivalent {@code CacheBuilderSpec}. The order
   * and form of this representation is not guaranteed, except that reparsing its output will
   * produce a {@code CacheBuilderSpec} equal to this instance.
   */
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
