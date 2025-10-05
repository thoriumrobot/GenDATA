// Source-based slice around line 130
// Method: <com.google.common.primitives.UnsignedLongs: String join(String,long)>


  /**
   * Returns a string containing the supplied unsigned {@code long} values separated by {@code
   * separator}. For example, {@code join("-", 1, 2, 3)} returns the string {@code "1-2-3"}.
   *
   * @param separator the text that should appear between consecutive values in the resulting string
   *     (but not at the start or end)
   * @param array an array of unsigned {@code long} values, possibly empty
   */
  public static String join(String separator, long... array) {
    checkNotNull(separator);
    if (array.length == 0) {
      return "";
    }

    // For pre-sizing a builder, just get the right order of magnitude
    StringBuilder builder = new StringBuilder(array.length * 5);
    builder.append(toString(array[0]));
    for (int i = 1; i < array.length; i++) {
      builder.append(separator).append(toString(array[i]));
