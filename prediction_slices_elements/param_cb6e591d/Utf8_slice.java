// Source-based slice around line 78
// Method: <com.google.common.base.Utf8: int encodedLengthGeneral(CharSequence,int)>


    if (utf8Length < utf16Length) {
      // Necessary and sufficient condition for overflow because of maximum 3x expansion
      throw new IllegalArgumentException(
          "UTF-8 length does not fit in int: " + (utf8Length + (1L << 32)));
    }
    return utf8Length;
  }

  private static int encodedLengthGeneral(CharSequence sequence, int start) {
    int utf16Length = sequence.length();
    int utf8Length = 0;
    for (int i = start; i < utf16Length; i++) {
      char c = sequence.charAt(i);
      if (c < 0x800) {
        utf8Length += (0x7f - c) >>> 31; // branch free!
      } else {
        utf8Length += 2;
        // We can't use Character.isSurrogate(c) here and below because of GWT.
        if (MIN_SURROGATE <= c && c <= MAX_SURROGATE) {
