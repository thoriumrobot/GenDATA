// Source-based slice around line 544
// Method: <com.google.common.base.CharMatcher: int indexIn(CharSequence)>

   * Returns the index of the first matching BMP character in a character sequence, or {@code -1} if
   * no matching character is present.
   *
   * <p>The default implementation iterates over the sequence in forward order calling {@link
   * #matches} for each character.
   *
   * @param sequence the character sequence to examine from the beginning
   * @return an index, or {@code -1} if no character matches
   */
  public int indexIn(CharSequence sequence) {
    return indexIn(sequence, 0);
  }

  /**
   * Returns the index of the first matching BMP character in a character sequence, starting from a
   * given position, or {@code -1} if no character matches after that position.
   *
   * <p>The default implementation iterates over the sequence in forward order, beginning at {@code
   * start}, calling {@link #matches} for each character.
   *
