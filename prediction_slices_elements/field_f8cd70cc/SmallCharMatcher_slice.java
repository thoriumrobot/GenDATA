// Source-based slice around line 42
// Method: com.google.common.base.SmallCharMatcher.C1

  private final long filter;

  private SmallCharMatcher(char[] table, long filter, boolean containsZero, String description) {
    super(description);
    this.table = table;
    this.filter = filter;
    this.containsZero = containsZero;
  }

  private static final int C1 = 0xcc9e2d51;
  private static final int C2 = 0x1b873593;

  /*
   * This method was rewritten in Java from an intermediate step of the Murmur hash function in
   * http://code.google.com/p/smhasher/source/browse/trunk/MurmurHash3.cpp, which contained the
   * following header:
   *
   * MurmurHash3 was written by Austin Appleby, and is placed in the public domain. The author
   * hereby disclaims copyright to this source code.
   */
