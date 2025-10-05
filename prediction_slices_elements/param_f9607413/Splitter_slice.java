// Source-based slice around line 421
// Method: <com.google.common.base.Splitter: Stream splitToStream(CharSequence)>

  /**
   * Splits {@code sequence} into string components and makes them available through an {@link
   * Stream}, which may be lazily evaluated. If you want an eagerly computed {@link List}, use
   * {@link #splitToList(CharSequence)}.
   *
   * @param sequence the sequence of characters to split
   * @return a stream over the segments split from the parameter
   * @since 28.2 (but only since 33.4.0 in the Android flavor)
   */
  public Stream<String> splitToStream(CharSequence sequence) {
    // Can't use Streams.stream() from base
    return StreamSupport.stream(split(sequence).spliterator(), false);
  }

  /**
   * Returns a {@code MapSplitter} which splits entries based on this splitter, and splits entries
   * into keys and values using the specified separator.
   *
   * @since 10.0
   */
