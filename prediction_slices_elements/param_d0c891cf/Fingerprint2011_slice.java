// Source-based slice around line 172
// Method: <com.google.common.hash.Fingerprint2011: long murmurHash64WithSeed(byte[],int,int,long)>

    c += rotateRight(a, 7);
    a += load64(bytes, offset + length - 16);
    long wf = a + z;
    long ws = b + rotateRight(a, 31) + c;
    long r = shiftMix((vf + ws) * K2 + (wf + vs) * K0);
    return shiftMix(r * K0 + vs) * K2;
  }

  @VisibleForTesting
  static long murmurHash64WithSeed(byte[] bytes, int offset, int length, long seed) {
    long mul = K3;
    int topBit = 0x7;

    int lengthAligned = length & ~topBit;
    int lengthRemainder = length & topBit;
    long hash = seed ^ (length * mul);

    for (int i = 0; i < lengthAligned; i += 8) {
      long loaded = load64(bytes, offset + i);
      long data = shiftMix(loaded * mul) * mul;
