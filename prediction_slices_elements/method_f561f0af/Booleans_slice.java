// Source-based slice around line 286
// Method: <com.google.common.primitives.Booleans: String join(String,boolean)>

  /**
   * Returns a string containing the supplied {@code boolean} values separated by {@code separator}.
   * For example, {@code join("-", false, true, false)} returns the string {@code
   * "false-true-false"}.
   *
   * @param separator the text that should appear between consecutive values in the resulting string
   *     (but not at the start or end)
   * @param array an array of {@code boolean} values, possibly empty
   */
  public static String join(String separator, boolean... array) {
    checkNotNull(separator);
    if (array.length == 0) {
      return "";
    }

    // For pre-sizing a builder, just get the right order of magnitude
    StringBuilder builder = new StringBuilder(array.length * 7);
    builder.append(array[0]);
    for (int i = 1; i < array.length; i++) {
      builder.append(separator).append(array[i]);
