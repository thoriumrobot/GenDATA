// Source-based slice around line 141
// Method: <com.google.common.cache.CacheBuilderSpec: CacheBuilderSpec parse(String)>

  private CacheBuilderSpec(String specification) {
    this.specification = specification;
  }

  /**
   * Creates a CacheBuilderSpec from a string.
   *
   * @param cacheBuilderSpecification the string form
   */
  public static CacheBuilderSpec parse(String cacheBuilderSpecification) {
    CacheBuilderSpec spec = new CacheBuilderSpec(cacheBuilderSpecification);
    if (!cacheBuilderSpecification.isEmpty()) {
      for (String keyValuePair : KEYS_SPLITTER.split(cacheBuilderSpecification)) {
        List<String> keyAndValue = ImmutableList.copyOf(KEY_VALUE_SPLITTER.split(keyValuePair));
        checkArgument(!keyAndValue.isEmpty(), "blank key-value pair");
        checkArgument(
            keyAndValue.size() <= 2,
            "key-value pair %s with more than one equals sign",
            keyValuePair);

