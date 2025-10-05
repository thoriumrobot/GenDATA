// Source-based slice around line 418
// Method: <com.google.common.base.CharMatcher: CharMatcher precomputedInternal()>

   * on {@link Platform} so that we can have different behavior in GWT.
   *
   * <p>This implementation tries to be smart in a number of ways. It recognizes cases where the
   * negation is cheaper to precompute than the matcher itself; it tries to build small hash tables
   * for matchers that only match a few characters, and so on. In the worst-case scenario, it
   * constructs an eight-kilobyte bit array and queries that. In many situations this produces a
   * matcher which is faster to query than the original.
   */
  @GwtIncompatible // SmallCharMatcher
  CharMatcher precomputedInternal() {
    BitSet table = new BitSet();
    setBits(table);
    int totalCharacters = table.cardinality();
    if (totalCharacters * 2 <= DISTINCT_CHARS) {
      return precomputedPositive(totalCharacters, table, toString());
    } else {
      // TODO(lowasser): is it worth it to worry about the last character of large matchers?
      table.flip(Character.MIN_VALUE, Character.MAX_VALUE + 1);
      int negatedCharacters = DISTINCT_CHARS - totalCharacters;
      String suffix = ".negate()";
