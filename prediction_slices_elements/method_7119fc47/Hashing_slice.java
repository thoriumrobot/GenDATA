// Source-based slice around line 51
// Method: <com.google.common.collect.Hashing: int smear(int)>


  /*
   * This method was rewritten in Java from an intermediate step of the Murmur hash function in
   * http://code.google.com/p/smhasher/source/browse/trunk/MurmurHash3.cpp, which contained the
   * following header:
   *
   * MurmurHash3 was written by Austin Appleby, and is placed in the public domain. The author
   * hereby disclaims copyright to this source code.
   */
  static int smear(int hashCode) {
    return (int) (C2 * Integer.rotateLeft((int) (hashCode * C1), 15));
  }

  static int smearedHash(@Nullable Object o) {
    return smear((o == null) ? 0 : o.hashCode());
  }

  private static final int MAX_TABLE_SIZE = Ints.MAX_POWER_OF_TWO;

  static int closedTableSize(int expectedEntries, double loadFactor) {
