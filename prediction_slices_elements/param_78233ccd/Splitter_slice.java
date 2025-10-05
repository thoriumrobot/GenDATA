// Source-based slice around line 465
// Method: <com.google.common.base.Splitter: MapSplitter withKeyValueSeparator(Splitter)>

   * String toSplit = " x -> y, z-> a ";
   * Splitter outerSplitter = Splitter.on(',').trimResults();
   * MapSplitter mapSplitter = outerSplitter.withKeyValueSeparator(Splitter.on("->"));
   * Map<String, String> result = mapSplitter.split(toSplit);
   * assertThat(result).isEqualTo(ImmutableMap.of("x ", " y", "z", " a"));
   * }
   *
   * @since 10.0
   */
  public MapSplitter withKeyValueSeparator(Splitter keyValueSplitter) {
    return new MapSplitter(this, keyValueSplitter);
  }

  /**
   * An object that splits strings into maps as {@code Splitter} splits iterables and lists. Like
   * {@code Splitter}, it is thread-safe and immutable. The common way to build instances is by
   * providing an additional {@linkplain Splitter#withKeyValueSeparator key-value separator} to
   * {@link Splitter}.
   *
   * @since 10.0
