// Source-based slice around line 40
// Method: com.google.common.collect.Hashing.C1

 */
@GwtCompatible
final class Hashing {
  private Hashing() {}

  /*
   * These should be ints, but we need to use longs to force GWT to do the multiplications with
   * enough precision.
   */
  private static final long C1 = 0xcc9e2d51;
  private static final long C2 = 0x1b873593;

  /*
   * This method was rewritten in Java from an intermediate step of the Murmur hash function in
   * http://code.google.com/p/smhasher/source/browse/trunk/MurmurHash3.cpp, which contained the
   * following header:
   *
   * MurmurHash3 was written by Austin Appleby, and is placed in the public domain. The author
   * hereby disclaims copyright to this source code.
   */
