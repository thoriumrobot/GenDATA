// Source-based slice around line 73
// Method: <com.google.common.base.SmallCharMatcher: int chooseTableSize(int)>

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
    }
    // Correct the size for open addressing to match desired load factor.
    // Round up to the next highest power of 2.
    int tableSize = Integer.highestOneBit(setSize - 1) << 1;
    while (tableSize * DESIRED_LOAD_FACTOR < setSize) {
      tableSize <<= 1;
    }
    return tableSize;
