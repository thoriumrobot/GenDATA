// Source-based slice around line 86
// Method: <com.google.common.base.SmallCharMatcher: CharMatcher from(BitSet,String)>

    // Correct the size for open addressing to match desired load factor.
    // Round up to the next highest power of 2.
    int tableSize = Integer.highestOneBit(setSize - 1) << 1;
    while (tableSize * DESIRED_LOAD_FACTOR < setSize) {
      tableSize <<= 1;
    }
    return tableSize;
  }

  static CharMatcher from(BitSet chars, String description) {
    // Compute the filter.
    long filter = 0;
    int size = chars.cardinality();
    boolean containsZero = chars.get(0);
    // Compute the hash table.
    char[] table = new char[chooseTableSize(size)];
    int mask = table.length - 1;
    for (int c = chars.nextSetBit(0); c != -1; c = chars.nextSetBit(c + 1)) {
      // Compute the filter at the same time.
      filter |= 1L << c;
