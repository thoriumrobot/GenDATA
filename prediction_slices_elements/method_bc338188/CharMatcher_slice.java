// Source-based slice around line 584
// Method: <com.google.common.base.CharMatcher: int lastIndexIn(CharSequence)>

   * Returns the index of the last matching BMP character in a character sequence, or {@code -1} if
   * no matching character is present.
   *
   * <p>The default implementation iterates over the sequence in reverse order calling {@link
   * #matches} for each character.
   *
   * @param sequence the character sequence to examine from the end
   * @return an index, or {@code -1} if no character matches
   */
  public int lastIndexIn(CharSequence sequence) {
    for (int i = sequence.length() - 1; i >= 0; i--) {
      if (matches(sequence.charAt(i))) {
        return i;
      }
    }
    return -1;
  }

  /**
   * Returns the number of matching {@code char}s found in a character sequence.
