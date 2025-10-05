// Source-based slice around line 57
// Method: <com.google.common.base.SmallCharMatcher: boolean checkFilter(int)>

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

  // Represents how tightly we can pack things, as a maximum.
  private static final double DESIRED_LOAD_FACTOR = 0.5;

  /**
