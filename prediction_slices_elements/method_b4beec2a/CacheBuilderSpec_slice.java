// Source-based slice around line 172
// Method: <com.google.common.cache.CacheBuilderSpec: CacheBuilder toCacheBuilder()>

  }

  /** Returns a CacheBuilderSpec that will prevent caching. */
  public static CacheBuilderSpec disableCaching() {
    // Maximum size of zero is one way to block caching
    return CacheBuilderSpec.parse("maximumSize=0");
  }

  /** Returns a CacheBuilder configured according to this instance's specification. */
  CacheBuilder<Object, Object> toCacheBuilder() {
    CacheBuilder<Object, Object> builder = CacheBuilder.newBuilder();
    if (initialCapacity != null) {
      builder.initialCapacity(initialCapacity);
    }
    if (maximumSize != null) {
      builder.maximumSize(maximumSize);
    }
    if (maximumWeight != null) {
      builder.maximumWeight(maximumWeight);
    }
