// Source-based slice around line 104
// Method: <com.google.common.hash.Murmur3_32HashFunction: int hashCode()>

  public boolean equals(@Nullable Object object) {
    if (object instanceof Murmur3_32HashFunction) {
      Murmur3_32HashFunction other = (Murmur3_32HashFunction) object;
      return seed == other.seed && supplementaryPlaneFix == other.supplementaryPlaneFix;
    }
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
