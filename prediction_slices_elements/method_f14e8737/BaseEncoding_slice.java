// Source-based slice around line 312
// Method: <com.google.common.io.BaseEncoding: BaseEncoding upperCase()>

  public abstract BaseEncoding withSeparator(String separator, int n);

  /**
   * Returns an encoding that behaves equivalently to this encoding, but encodes and decodes with
   * uppercase letters. Padding and separator characters remain in their original case.
   *
   * @throws IllegalStateException if the alphabet used by this encoding contains mixed upper- and
   *     lower-case characters
   */
  public abstract BaseEncoding upperCase();

  /**
   * Returns an encoding that behaves equivalently to this encoding, but encodes and decodes with
   * lowercase letters. Padding and separator characters remain in their original case.
   *
   * @throws IllegalStateException if the alphabet used by this encoding contains mixed upper- and
   *     lower-case characters
   */
  public abstract BaseEncoding lowerCase();

