// Source-based slice around line 730
// Method: <com.google.common.primitives.Doubles: java fpPattern()>

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
     */
    String decimal = "(?:\\d+#(?:\\.\\d*#)?|\\.\\d+#)";
    String completeDec = decimal + "(?:[eE][+-]?\\d+#)?[fFdD]?";
    String hex = "(?:[0-9a-fA-F]+#(?:\\.[0-9a-fA-F]*#)?|\\.[0-9a-fA-F]+#)";
    String completeHex = "0[xX]" + hex + "[pP][+-]?\\d+#[fFdD]?";
