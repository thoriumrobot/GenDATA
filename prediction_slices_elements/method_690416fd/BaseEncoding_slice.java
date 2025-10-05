// Source-based slice around line 303
// Method: <com.google.common.io.BaseEncoding: BaseEncoding withSeparator(String,int)>

  /**
   * Returns an encoding that behaves equivalently to this encoding, but adds a separator string
   * after every {@code n} characters. Any occurrences of any characters that occur in the separator
   * are skipped over in decoding.
   *
   * @throws IllegalArgumentException if any alphabet or padding characters appear in the separator
   *     string, or if {@code n <= 0}
   * @throws UnsupportedOperationException if this encoding already uses a separator
   */
  public abstract BaseEncoding withSeparator(String separator, int n);

  /**
   * Returns an encoding that behaves equivalently to this encoding, but encodes and decodes with
   * uppercase letters. Padding and separator characters remain in their original case.
   *
   * @throws IllegalStateException if the alphabet used by this encoding contains mixed upper- and
   *     lower-case characters
   */
  public abstract BaseEncoding upperCase();

