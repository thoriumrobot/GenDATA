// Source-based slice around line 53
// Method: <com.google.common.base.SmallCharMatcher: int smear(int)>


  /*
   * This method was rewritten in Java from an intermediate step of the Murmur hash function in
   * http://code.google.com/p/smhasher/source/browse/trunk/MurmurHash3.cpp, which contained the
   * following header:
   *
   * MurmurHash3 was written by Austin Appleby, and is placed in the public domain. The author
   * hereby disclaims copyright to this source code.
   */
  static int smear(int hashCode) {
    return C2 * Integer.rotateLeft(hashCode * C1, 15);
  }

  private boolean checkFilter(int c) {
    return ((filter >> c) & 1) == 1;
  }

  // This is all essentially copied from ImmutableSet, but we have to duplicate because
  // of dependencies.

