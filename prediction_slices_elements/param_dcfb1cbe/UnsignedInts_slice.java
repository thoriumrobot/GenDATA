// Source-based slice around line 70
// Method: <com.google.common.primitives.UnsignedInts: int compare(int,int)>

   * <p><b>Note:</b> this method is now unnecessary and should be treated as deprecated; use the
   * equivalent {@link Integer#compareUnsigned(int, int)} method instead.
   *
   * @param a the first unsigned {@code int} to compare
   * @param b the second unsigned {@code int} to compare
   * @return a negative value if {@code a} is less than {@code b}; a positive value if {@code a} is
   *     greater than {@code b}; or zero if they are equal
   */
  @SuppressWarnings("InlineMeInliner") // Integer.compare unavailable under GWT+J2CL
  public static int compare(int a, int b) {
    return Ints.compare(flip(a), flip(b));
  }

  /**
   * Returns the value of the given {@code int} as a {@code long}, when treated as unsigned.
   *
   * <p><b>Java 8+ users:</b> use {@link Integer#toUnsignedLong(int)} instead.
   */
  public static long toLong(int value) {
    return value & INT_MASK;
