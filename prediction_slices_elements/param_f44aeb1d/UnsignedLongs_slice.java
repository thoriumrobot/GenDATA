// Source-based slice around line 78
// Method: <com.google.common.primitives.UnsignedLongs: int compare(long,long)>

   * <p><b>Note:</b> this method is now unnecessary and should be treated as deprecated; use the
   * equivalent {@link Long#compareUnsigned(long, long)} method instead.
   *
   * @param a the first unsigned {@code long} to compare
   * @param b the second unsigned {@code long} to compare
   * @return a negative value if {@code a} is less than {@code b}; a positive value if {@code a} is
   *     greater than {@code b}; or zero if they are equal
   */
  @SuppressWarnings("InlineMeInliner") // Integer.compare unavailable under GWT+J2CL
  public static int compare(long a, long b) {
    return Longs.compare(flip(a), flip(b));
  }

  /**
   * Returns the least value present in {@code array}, treating values as unsigned.
   *
   * @param array a <i>nonempty</i> array of unsigned {@code long} values
   * @return the value present in {@code array} that is less than or equal to every other value in
   *     the array according to {@link #compare}
   * @throws IllegalArgumentException if {@code array} is empty
