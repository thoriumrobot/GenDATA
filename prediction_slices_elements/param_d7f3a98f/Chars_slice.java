// Source-based slice around line 370
// Method: <com.google.common.primitives.Chars: String join(String,char)>


  /**
   * Returns a string containing the supplied {@code char} values separated by {@code separator}.
   * For example, {@code join("-", '1', '2', '3')} returns the string {@code "1-2-3"}.
   *
   * @param separator the text that should appear between consecutive values in the resulting string
   *     (but not at the start or end)
   * @param array an array of {@code char} values, possibly empty
   */
  public static String join(String separator, char... array) {
    checkNotNull(separator);
    int len = array.length;
    if (len == 0) {
      return "";
    }

    StringBuilder builder = new StringBuilder(len + separator.length() * (len - 1));
    builder.append(array[0]);
    for (int i = 1; i < len; i++) {
      builder.append(separator).append(array[i]);
