// Source-based slice around line 69
// Method: <com.google.common.hash.Murmur3_128HashFunction: String toString()>

    return 128;
  }

  @Override
  public Hasher newHasher() {
    return new Murmur3_128Hasher(seed);
  }

  @Override
  public String toString() {
    return "Hashing.murmur3_128(" + seed + ")";
  }

  @Override
  public boolean equals(@Nullable Object object) {
    if (object instanceof Murmur3_128HashFunction) {
      Murmur3_128HashFunction other = (Murmur3_128HashFunction) object;
      return seed == other.seed;
    }
    return false;
