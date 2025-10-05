// Source-based slice around line 564
// Method: <com.google.common.primitives.Ints: void rotate(int[],int,int,int)>

   * Collections.rotate(Ints.asList(array).subList(fromIndex, toIndex), distance)}, but is
   * considerably faster and avoids allocations and garbage collection.
   *
   * <p>The provided "distance" may be negative, which will rotate left.
   *
   * @throws IndexOutOfBoundsException if {@code fromIndex < 0}, {@code toIndex > array.length}, or
   *     {@code toIndex > fromIndex}
   * @since 32.0.0
   */
  public static void rotate(int[] array, int distance, int fromIndex, int toIndex) {
    // There are several well-known algorithms for rotating part of an array (or, equivalently,
    // exchanging two blocks of memory). This classic text by Gries and Mills mentions several:
    // https://ecommons.cornell.edu/bitstream/handle/1813/6292/81-452.pdf.
    // (1) "Reversal", the one we have here.
    // (2) "Dolphin". If we're rotating an array a of size n by a distance of d, then element a[0]
    //     ends up at a[d], which in turn ends up at a[2d], and so on until we get back to a[0].
    //     (All indices taken mod n.) If d and n are mutually prime, all elements will have been
    //     moved at that point. Otherwise, we can rotate the cycle a[1], a[1 + d], a[1 + 2d], etc,
    //     then a[2] etc, and so on until we have rotated all elements. There are gcd(d, n) cycles
    //     in all.
