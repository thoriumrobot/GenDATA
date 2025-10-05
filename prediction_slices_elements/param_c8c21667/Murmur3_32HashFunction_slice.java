// Source-based slice around line 109
// Method: <com.google.common.hash.Murmur3_32HashFunction: HashCode hashInt(int)>

    return false;
  }

  @Override
  public int hashCode() {
    return getClass().hashCode() ^ seed;
  }

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
