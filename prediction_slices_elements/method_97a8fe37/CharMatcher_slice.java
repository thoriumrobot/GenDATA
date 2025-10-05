// Source-based slice around line 510
// Method: <com.google.common.base.CharMatcher: boolean matchesAllOf(CharSequence)>

   * Returns {@code true} if a character sequence contains only matching BMP characters.
   *
   * <p>The default implementation iterates over the sequence, invoking {@link #matches} for each
   * character, until this returns {@code false} or the end is reached.
   *
   * @param sequence the character sequence to examine, possibly empty
   * @return {@code true} if this matcher matches every character in the sequence, including when
   *     the sequence is empty
   */
  public boolean matchesAllOf(CharSequence sequence) {
    for (int i = sequence.length() - 1; i >= 0; i--) {
      if (!matches(sequence.charAt(i))) {
        return false;
      }
    }
    return true;
  }

  /**
   * Returns {@code true} if a character sequence contains no matching BMP characters. Equivalent to
