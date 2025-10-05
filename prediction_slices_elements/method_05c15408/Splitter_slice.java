// Source-based slice around line 442
// Method: <com.google.common.base.Splitter: MapSplitter withKeyValueSeparator(char)>

    return withKeyValueSeparator(on(separator));
  }

  /**
   * Returns a {@code MapSplitter} which splits entries based on this splitter, and splits entries
   * into keys and values using the specified separator.
   *
   * @since 14.0
   */
  public MapSplitter withKeyValueSeparator(char separator) {
    return withKeyValueSeparator(on(separator));
  }

  /**
   * Returns a {@code MapSplitter} which splits entries based on this splitter, and splits entries
   * into keys and values using the specified key-value splitter.
   *
   * <p>Note: Any configuration option configured on this splitter, such as {@link #trimResults},
   * does not change the behavior of the {@code keyValueSplitter}.
   *
