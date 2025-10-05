// Source-based slice around line 366
// Method: <com.google.common.primitives.Doubles: String join(String,double)>

   * {@code join("-", 1.0, 2.0, 3.0)} returns the string {@code "1.0-2.0-3.0"}.
   *
   * <p>Note that {@link Double#toString(double)} formats {@code double} differently in GWT
   * sometimes. In the previous example, it returns the string {@code "1-2-3"}.
   *
   * @param separator the text that should appear between consecutive values in the resulting string
   *     (but not at the start or end)
   * @param array an array of {@code double} values, possibly empty
   */
  public static String join(String separator, double... array) {
    checkNotNull(separator);
    if (array.length == 0) {
      return "";
    }

    // For pre-sizing a builder, just get the right order of magnitude
    StringBuilder builder = new StringBuilder(array.length * 12);
    builder.append(array[0]);
    for (int i = 1; i < array.length; i++) {
      builder.append(separator).append(array[i]);
