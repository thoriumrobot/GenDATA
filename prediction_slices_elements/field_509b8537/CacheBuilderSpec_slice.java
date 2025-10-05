// Source-based slice around line 93
// Method: com.google.common.cache.CacheBuilderSpec.KEYS_SPLITTER

@SuppressWarnings("GoodTime") // lots of violations (nanosecond math)
@GwtIncompatible
public final class CacheBuilderSpec {
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
