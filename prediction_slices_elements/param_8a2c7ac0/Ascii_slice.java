// Source-based slice around line 601
// Method: <com.google.common.base.Ascii: boolean equalsIgnoreCase(CharSequence,CharSequence)>

   * </ul>
   *
   * <p>due to case-folding of some non-ASCII characters (which does not occur in {@link
   * String#equalsIgnoreCase}). However in almost all cases that ASCII strings are used, the author
   * probably wanted the behavior provided by this method rather than the subtle and sometimes
   * surprising behavior of {@code toUpperCase()} and {@code toLowerCase()}.
   *
   * @since 16.0
   */
  public static boolean equalsIgnoreCase(CharSequence s1, CharSequence s2) {
    // Calling length() is the null pointer check (so do it before we can exit early).
    int length = s1.length();
    if (s1 == s2) {
      return true;
    }
    if (length != s2.length()) {
      return false;
    }
    for (int i = 0; i < length; i++) {
      char c1 = s1.charAt(i);
