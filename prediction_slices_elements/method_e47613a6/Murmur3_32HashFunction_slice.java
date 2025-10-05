// Source-based slice around line 90
// Method: <com.google.common.hash.Murmur3_32HashFunction: String toString()>

    return 32;
  }

  @Override
  public Hasher newHasher() {
    return new Murmur3_32Hasher(seed);
  }

  @Override
  public String toString() {
    return "Hashing.murmur3_32(" + seed + ")";
  }

  @Override
  public boolean equals(@Nullable Object object) {
    if (object instanceof Murmur3_32HashFunction) {
      Murmur3_32HashFunction other = (Murmur3_32HashFunction) object;
      return seed == other.seed && supplementaryPlaneFix == other.supplementaryPlaneFix;
    }
    return false;
