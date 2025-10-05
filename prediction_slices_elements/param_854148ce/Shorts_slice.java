// Source-based slice around line 421
// Method: <com.google.common.primitives.Shorts: String join(String,short)>

  /**
   * Returns a string containing the supplied {@code short} values separated by {@code separator}.
   * For example, {@code join("-", (short) 1, (short) 2, (short) 3)} returns the string {@code
   * "1-2-3"}.
   *
   * @param separator the text that should appear between consecutive values in the resulting string
   *     (but not at the start or end)
   * @param array an array of {@code short} values, possibly empty
   */
  public static String join(String separator, short... array) {
    checkNotNull(separator);
    if (array.length == 0) {
      return "";
    }

    // For pre-sizing a builder, just get the right order of magnitude
    StringBuilder builder = new StringBuilder(array.length * 6);
    builder.append(array[0]);
    for (int i = 1; i < array.length; i++) {
      builder.append(separator).append(array[i]);
