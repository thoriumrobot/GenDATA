// Source-based slice around line 530
// Method: <com.google.common.base.CharMatcher: boolean matchesNoneOf(CharSequence)>

   * {@code !matchesAnyOf(sequence)}.
   *
   * <p>The default implementation iterates over the sequence, invoking {@link #matches} for each
   * character, until this returns {@code true} or the end is reached.
   *
   * @param sequence the character sequence to examine, possibly empty
   * @return {@code true} if this matcher matches no characters in the sequence, including when the
   *     sequence is empty
   */
  public boolean matchesNoneOf(CharSequence sequence) {
    return indexIn(sequence) == -1;
  }

  /**
   * Returns the index of the first matching BMP character in a character sequence, or {@code -1} if
   * no matching character is present.
   *
   * <p>The default implementation iterates over the sequence in forward order calling {@link
   * #matches} for each character.
   *
