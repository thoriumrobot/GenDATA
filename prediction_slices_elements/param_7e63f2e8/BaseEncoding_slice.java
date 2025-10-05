// Source-based slice around line 292
// Method: <com.google.common.io.BaseEncoding: BaseEncoding withPadChar(char)>

  public abstract BaseEncoding omitPadding();

  /**
   * Returns an encoding that behaves equivalently to this encoding, but uses an alternate character
   * for padding.
   *
   * @throws IllegalArgumentException if this padding character is already used in the alphabet or a
   *     separator
   */
  public abstract BaseEncoding withPadChar(char padChar);

  /**
   * Returns an encoding that behaves equivalently to this encoding, but adds a separator string
   * after every {@code n} characters. Any occurrences of any characters that occur in the separator
   * are skipped over in decoding.
   *
   * @throws IllegalArgumentException if any alphabet or padding characters appear in the separator
   *     string, or if {@code n <= 0}
   * @throws UnsupportedOperationException if this encoding already uses a separator
   */
