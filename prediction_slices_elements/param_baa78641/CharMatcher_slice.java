// Source-based slice around line 1717
// Method: <com.google.common.base.CharMatcher: CharMatcher isEither(char,char)>

      return is(match);
    }

    @Override
    public String toString() {
      return "CharMatcher.isNot('" + showCharacter(match) + "')";
    }
  }

  private static CharMatcher.IsEither isEither(char c1, char c2) {
    return new CharMatcher.IsEither(c1, c2);
  }

  /** Implementation of {@link #anyOf(CharSequence)} for exactly two characters. */
  private static final class IsEither extends FastMatcher {

    private final char match1;
    private final char match2;

    IsEither(char match1, char match2) {
