// Source-based slice around line 725
// Method: com.google.common.primitives.Doubles.FLOATING_POINT_PATTERN

  }

  /**
   * This is adapted from the regex suggested by {@link Double#valueOf(String)} for prevalidating
   * inputs. All valid inputs must pass this regex, but it's semantically fine if not all inputs
   * that pass this regex are valid -- only a performance hit is incurred, not a semantics bug.
   */
  @GwtIncompatible // regular expressions
  static final
  java.util.regex.Pattern
      FLOATING_POINT_PATTERN = fpPattern();

  @GwtIncompatible // regular expressions
  private static
  java.util.regex.Pattern
      fpPattern() {
    /*
     * We use # instead of * for possessive quantifiers. This lets us strip them out when building
     * the regex for RE2 (which doesn't support them) but leave them in when building it for
     * java.util.regex (where we want them in order to avoid catastrophic backtracking).
