// Source-based slice around line 244
// Method: <com.google.common.hash.Murmur3_32HashFunction: int mixK1(int)>

    }
    h1 ^= mixK1(k1);
    return fmix(h1, len);
  }

  private static int getIntLittleEndian(byte[] input, int offset) {
    return Ints.fromBytes(input[offset + 3], input[offset + 2], input[offset + 1], input[offset]);
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
