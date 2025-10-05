// Source-based slice around line 251
// Method: <com.google.common.hash.Murmur3_32HashFunction: int mixH1(int,int)>

  }

  private static int mixK1(int k1) {
    k1 *= C1;
    k1 = Integer.rotateLeft(k1, 15);
    k1 *= C2;
    return k1;
  }

  private static int mixH1(int h1, int k1) {
    h1 ^= k1;
    h1 = Integer.rotateLeft(h1, 13);
    h1 = h1 * 5 + 0xe6546b64;
    return h1;
  }

  // Finalization mix - force all bits of a hash block to avalanche
  private static HashCode fmix(int h1, int length) {
    h1 ^= length;
    h1 ^= h1 >>> 16;
