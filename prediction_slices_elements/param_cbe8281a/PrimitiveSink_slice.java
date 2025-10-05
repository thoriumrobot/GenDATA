// Source-based slice around line 100
// Method: <com.google.common.hash.PrimitiveSink: PrimitiveSink putChar(char)>

  @CanIgnoreReturnValue
  PrimitiveSink putDouble(double d);

  /** Puts a boolean into this sink. */
  @CanIgnoreReturnValue
  PrimitiveSink putBoolean(boolean b);

  /** Puts a character into this sink. */
  @CanIgnoreReturnValue
  PrimitiveSink putChar(char c);

  /**
   * Puts each 16-bit code unit from the {@link CharSequence} into this sink.
   *
   * <p><b>Warning:</b> This method will produce different output than most other languages do when
   * running on the equivalent input. For cross-language compatibility, use {@link #putString},
   * usually with a charset of UTF-8. For other use cases, use {@code putUnencodedChars}.
   *
   * @since 15.0 (since 11.0 as putString(CharSequence))
   */
