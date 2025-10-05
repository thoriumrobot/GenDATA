// Source-based slice around line 112
// Method: <com.google.common.hash.PrimitiveSink: PrimitiveSink putUnencodedChars(CharSequence)>

   * Puts each 16-bit code unit from the {@link CharSequence} into this sink.
   *
   * <p><b>Warning:</b> This method will produce different output than most other languages do when
   * running on the equivalent input. For cross-language compatibility, use {@link #putString},
   * usually with a charset of UTF-8. For other use cases, use {@code putUnencodedChars}.
   *
   * @since 15.0 (since 11.0 as putString(CharSequence))
   */
  @CanIgnoreReturnValue
  PrimitiveSink putUnencodedChars(CharSequence charSequence);

  /**
   * Puts a string into this sink using the given charset.
   *
   * <p><b>Warning:</b> This method, which reencodes the input before processing it, is useful only
   * for cross-language compatibility. For other use cases, prefer {@link #putUnencodedChars}, which
   * is faster, produces the same output across Java releases, and processes every {@code char} in
   * the input, even if some are invalid.
   */
  @CanIgnoreReturnValue
