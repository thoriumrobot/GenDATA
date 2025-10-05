// Source-based slice around line 117
// Method: <com.google.common.hash.Murmur3_32HashFunction: HashCode hashLong(long)>

  @Override
  public HashCode hashInt(int input) {
    int k1 = mixK1(input);
    int h1 = mixH1(seed, k1);

    return fmix(h1, Ints.BYTES);
  }

  @Override
  public HashCode hashLong(long input) {
    int low = (int) input;
    int high = (int) (input >>> 32);

    int k1 = mixK1(low);
    int h1 = mixH1(seed, k1);

    k1 = mixK1(high);
    h1 = mixH1(h1, k1);

    return fmix(h1, Longs.BYTES);
