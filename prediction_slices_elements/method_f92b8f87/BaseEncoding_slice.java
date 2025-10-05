// Source-based slice around line 321
// Method: <com.google.common.io.BaseEncoding: BaseEncoding lowerCase()>

  public abstract BaseEncoding upperCase();

  /**
   * Returns an encoding that behaves equivalently to this encoding, but encodes and decodes with
   * lowercase letters. Padding and separator characters remain in their original case.
   *
   * @throws IllegalStateException if the alphabet used by this encoding contains mixed upper- and
   *     lower-case characters
   */
  public abstract BaseEncoding lowerCase();

  /**
   * Returns an encoding that behaves equivalently to this encoding, but decodes letters without
   * regard to case.
   *
   * @throws IllegalStateException if the alphabet used by this encoding contains mixed upper- and
   *     lower-case characters
   * @since 32.0.0
   */
  public abstract BaseEncoding ignoreCase();
