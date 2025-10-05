// Source-based slice around line 240
// Method: <com.google.common.hash.Murmur3_32HashFunction: int getIntLittleEndian(byte[],int)>


    int k1 = 0;
    for (int shift = 0; i < len; i++, shift += 8) {
      k1 ^= toUnsignedInt(input[off + i]) << shift;
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

