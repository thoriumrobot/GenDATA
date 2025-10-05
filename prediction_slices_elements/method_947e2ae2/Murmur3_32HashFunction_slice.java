// Source-based slice around line 85
// Method: <com.google.common.hash.Murmur3_32HashFunction: Hasher newHasher()>

    this.supplementaryPlaneFix = supplementaryPlaneFix;
  }

  @Override
  public int bits() {
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
