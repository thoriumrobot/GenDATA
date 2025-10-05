// Source-based slice around line 136
// Method: <com.google.common.base.CaseFormat: String convert(CaseFormat,String)>

   * the behavior of this method is undefined but we make a reasonable effort at converting anyway.
   */
  public final String to(CaseFormat format, String str) {
    checkNotNull(format);
    checkNotNull(str);
    return (format == this) ? str : convert(format, str);
  }

  /** Enum values can override for performance reasons. */
  String convert(CaseFormat format, String s) {
    // deal with camel conversion
    StringBuilder out = null;
    int i = 0;
    int j = -1;
    while ((j = wordBoundary.indexIn(s, ++j)) != -1) {
      if (i == 0) {
        // include some extra space for separators
        out = new StringBuilder(s.length() + 4 * format.wordSeparator.length());
        out.append(format.normalizeFirstWord(s.substring(i, j)));
      } else {
