// Source-based slice around line 64
// Method: <com.google.common.hash.Murmur3_128HashFunction: Hasher newHasher()>

    this.seed = seed;
  }

  @Override
  public int bits() {
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
