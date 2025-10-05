// Source-based slice around line 123
// Method: <com.google.common.hash.PrimitiveSink: PrimitiveSink putString(CharSequence,Charset)>

  /**
   * Puts a string into this sink using the given charset.
   *
   * <p><b>Warning:</b> This method, which reencodes the input before processing it, is useful only
   * for cross-language compatibility. For other use cases, prefer {@link #putUnencodedChars}, which
   * is faster, produces the same output across Java releases, and processes every {@code char} in
   * the input, even if some are invalid.
   */
  @CanIgnoreReturnValue
  PrimitiveSink putString(CharSequence charSequence, Charset charset);
}
