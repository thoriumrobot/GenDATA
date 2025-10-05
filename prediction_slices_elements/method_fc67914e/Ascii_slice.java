// Source-based slice around line 550
// Method: <com.google.common.base.Ascii: String truncate(CharSequence,int,String)>

   *       into account
   *   <li>the appropriate truncation indicator may be locale-dependent
   *   <li>it is safe to use non-ASCII characters in the truncation indicator
   * </ul>
   *
   * @throws IllegalArgumentException if {@code maxLength} is less than the length of {@code
   *     truncationIndicator}
   * @since 16.0
   */
  public static String truncate(CharSequence seq, int maxLength, String truncationIndicator) {
    checkNotNull(seq);

    // length to truncate the sequence to, not including the truncation indicator
    int truncationLength = maxLength - truncationIndicator.length();

    // in this worst case, this allows a maxLength equal to the length of the truncationIndicator,
    // meaning that a string will be truncated to just the truncation indicator itself
    checkArgument(
        truncationLength >= 0,
        "maxLength (%s) must be >= length of the truncation indicator (%s)",
