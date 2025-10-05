// Source-based slice around line 576
// Method: <com.google.common.collect.Range: Range gap(Range)>

   * <p>The gap exists if and only if the two ranges are either disconnected or immediately adjacent
   * (any intersection must be an empty range).
   *
   * <p>The gap operation is commutative.
   *
   * @throws IllegalArgumentException if this range and {@code otherRange} have a nonempty
   *     intersection
   * @since 27.0
   */
  public Range<C> gap(Range<C> otherRange) {
    /*
     * For an explanation of the basic principle behind this check, see
     * https://stackoverflow.com/a/35754308/28465
     *
     * In that explanation's notation, our `overlap` check would be `x1 < y2 && y1 < x2`. We've
     * flipped one part of the check so that we're using "less than" in both cases (rather than a
     * mix of "less than" and "greater than"). We've also switched to "strictly less than" rather
     * than "less than or equal to" because of *handwave* the difference between "endpoints of
     * inclusive ranges" and "Cuts."
     */
