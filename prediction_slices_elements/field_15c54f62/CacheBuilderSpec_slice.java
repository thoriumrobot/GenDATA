// Source-based slice around line 96
// Method: com.google.common.cache.CacheBuilderSpec.KEY_VALUE_SPLITTER

  /** Parses a single value. */
  private interface ValueParser {
    void parse(CacheBuilderSpec spec, String key, @Nullable String value);
  }

  /** Splits each key-value pair. */
  private static final Splitter KEYS_SPLITTER = Splitter.on(',').trimResults();

  /** Splits the key from the value. */
  private static final Splitter KEY_VALUE_SPLITTER = Splitter.on('=').trimResults();

  /** Map of names to ValueParser. */
  private static final ImmutableMap<String, ValueParser> VALUE_PARSERS =
      ImmutableMap.<String, ValueParser>builder()
          .put("initialCapacity", new InitialCapacityParser())
          .put("maximumSize", new MaximumSizeParser())
          .put("maximumWeight", new MaximumWeightParser())
          .put("concurrencyLevel", new ConcurrencyLevelParser())
          .put("weakKeys", new KeyStrengthParser(Strength.WEAK))
          .put("softValues", new ValueStrengthParser(Strength.SOFT))
