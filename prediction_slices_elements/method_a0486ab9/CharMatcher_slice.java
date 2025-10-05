// Source-based slice around line 448
// Method: <com.google.common.base.CharMatcher: CharMatcher precomputedPositive(int,BitSet,String)>

        }
      };
    }
  }

  /**
   * Helper method for {@link #precomputedInternal} that doesn't test if the negation is cheaper.
   */
  @GwtIncompatible // SmallCharMatcher
  private static CharMatcher precomputedPositive(
      int totalCharacters, BitSet table, String description) {
    switch (totalCharacters) {
      case 0:
        return none();
      case 1:
        return is((char) table.nextSetBit(0));
      case 2:
        char c1 = (char) table.nextSetBit(0);
        char c2 = (char) table.nextSetBit(c1 + 1);
        return isEither(c1, c2);
