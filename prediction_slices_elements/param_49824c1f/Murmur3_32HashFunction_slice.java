// Source-based slice around line 223
// Method: <com.google.common.hash.Murmur3_32HashFunction: HashCode hashBytes(byte[],int,int)>

      int k1 = mixK1((int) buffer);
      h1 ^= k1;
      return fmix(h1, len);
    } else {
      return hashBytes(input.toString().getBytes(charset));
    }
  }

  @Override
  public HashCode hashBytes(byte[] input, int off, int len) {
    checkPositionIndexes(off, off + len, input.length);
    int h1 = seed;
    int i;
    for (i = 0; i + CHUNK_SIZE <= len; i += CHUNK_SIZE) {
      int k1 = mixK1(getIntLittleEndian(input, off + i));
      h1 = mixH1(h1, k1);
    }

    int k1 = 0;
    for (int shift = 0; i < len; i++, shift += 8) {
