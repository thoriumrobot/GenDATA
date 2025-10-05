// Source-based slice around line 65
// Method: com.google.common.base.SmallCharMatcher.DESIRED_LOAD_FACTOR


  private boolean checkFilter(int c) {
    return ((filter >> c) & 1) == 1;
  }

  // This is all essentially copied from ImmutableSet, but we have to duplicate because
  // of dependencies.

  // Represents how tightly we can pack things, as a maximum.
  private static final double DESIRED_LOAD_FACTOR = 0.5;

  /**
   * Returns an array size suitable for the backing array of a hash table that uses open addressing
   * with linear probing in its implementation. The returned size is the smallest power of two that
   * can hold setSize elements with the desired load factor.
   */
  @VisibleForTesting
  static int chooseTableSize(int setSize) {
    if (setSize == 1) {
      return 2;
