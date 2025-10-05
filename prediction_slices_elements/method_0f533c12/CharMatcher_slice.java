// Source-based slice around line 496
// Method: <com.google.common.base.CharMatcher: boolean matchesAnyOf(CharSequence)>

   * Equivalent to {@code !matchesNoneOf(sequence)}.
   *
   * <p>The default implementation iterates over the sequence, invoking {@link #matches} for each
   * character, until this returns {@code true} or the end is reached.
   *
   * @param sequence the character sequence to examine, possibly empty
   * @return {@code true} if this matcher matches at least one character in the sequence
   * @since 8.0
   */
  public boolean matchesAnyOf(CharSequence sequence) {
    return !matchesNoneOf(sequence);
  }

  /**
   * Returns {@code true} if a character sequence contains only matching BMP characters.
   *
   * <p>The default implementation iterates over the sequence, invoking {@link #matches} for each
   * character, until this returns {@code false} or the end is reached.
   *
   * @param sequence the character sequence to examine, possibly empty
