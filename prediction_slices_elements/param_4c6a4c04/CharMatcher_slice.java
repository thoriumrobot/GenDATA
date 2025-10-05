// Source-based slice around line 598
// Method: <com.google.common.base.CharMatcher: int countIn(CharSequence)>

    }
    return -1;
  }

  /**
   * Returns the number of matching {@code char}s found in a character sequence.
   *
   * <p>Counts 2 per supplementary character, such as for {@link #whitespace}().{@link #negate}().
   */
  public int countIn(CharSequence sequence) {
    int count = 0;
    for (int i = 0; i < sequence.length(); i++) {
      if (matches(sequence.charAt(i))) {
        count++;
      }
    }
    return count;
  }

  /**
