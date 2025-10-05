// Source-based slice around line 432
// Method: <com.google.common.base.Splitter: MapSplitter withKeyValueSeparator(String)>

    return StreamSupport.stream(split(sequence).spliterator(), false);
  }

  /**
   * Returns a {@code MapSplitter} which splits entries based on this splitter, and splits entries
   * into keys and values using the specified separator.
   *
   * @since 10.0
   */
  public MapSplitter withKeyValueSeparator(String separator) {
    return withKeyValueSeparator(on(separator));
  }

  /**
   * Returns a {@code MapSplitter} which splits entries based on this splitter, and splits entries
   * into keys and values using the specified separator.
   *
   * @since 14.0
   */
  public MapSplitter withKeyValueSeparator(char separator) {
