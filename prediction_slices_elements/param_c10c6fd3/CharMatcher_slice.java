// Source-based slice around line 563
// Method: <com.google.common.base.CharMatcher: int indexIn(CharSequence,int)>

   *
   * @param sequence the character sequence to examine
   * @param start the first index to examine; must be nonnegative and no greater than {@code
   *     sequence.length()}
   * @return the index of the first matching character, guaranteed to be no less than {@code start},
   *     or {@code -1} if no character matches
   * @throws IndexOutOfBoundsException if start is negative or greater than {@code
   *     sequence.length()}
   */
  public int indexIn(CharSequence sequence, int start) {
    int length = sequence.length();
    checkPositionIndex(start, length);
    for (int i = start; i < length; i++) {
      if (matches(sequence.charAt(i))) {
        return i;
      }
    }
    return -1;
  }

