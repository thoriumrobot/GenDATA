// Source-based slice around line 59
// Method: com.google.common.collect.Hashing.MAX_TABLE_SIZE

   */
  static int smear(int hashCode) {
    return (int) (C2 * Integer.rotateLeft((int) (hashCode * C1), 15));
  }

  static int smearedHash(@Nullable Object o) {
    return smear((o == null) ? 0 : o.hashCode());
  }

  private static final int MAX_TABLE_SIZE = Ints.MAX_POWER_OF_TWO;

  static int closedTableSize(int expectedEntries, double loadFactor) {
    // Get the recommended table size.
    // Round down to the nearest power of 2.
    expectedEntries = max(expectedEntries, 2);
    int tableSize = Integer.highestOneBit(expectedEntries);
    // Check to make sure that we will not exceed the maximum load factor.
    if (expectedEntries > (int) (loadFactor * tableSize)) {
      tableSize <<= 1;
      return (tableSize > 0) ? tableSize : MAX_TABLE_SIZE;
