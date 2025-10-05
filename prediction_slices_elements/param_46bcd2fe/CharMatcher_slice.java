// Source-based slice around line 467
// Method: <com.google.common.base.CharMatcher: boolean isSmall(int,int)>

        return isEither(c1, c2);
      default:
        return isSmall(totalCharacters, table.length())
            ? SmallCharMatcher.from(table, description)
            : new BitSetMatcher(table, description);
    }
  }

  @GwtIncompatible // SmallCharMatcher
  private static boolean isSmall(int totalCharacters, int tableLength) {
    return totalCharacters <= SmallCharMatcher.MAX_SIZE
        && tableLength > (totalCharacters * 4 * Character.SIZE);
    // err on the side of BitSetMatcher
  }

  /** Sets bits in {@code table} matched by this matcher. */
  @GwtIncompatible // used only from other GwtIncompatible code
  void setBits(BitSet table) {
    for (int c = Character.MAX_VALUE; c >= Character.MIN_VALUE; c--) {
      if (matches((char) c)) {
