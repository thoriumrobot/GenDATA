// Source-based slice around line 80
// Method: <com.google.common.hash.Murmur3_32HashFunction: int bits()>

  private final int seed;
  private final boolean supplementaryPlaneFix;

  Murmur3_32HashFunction(int seed, boolean supplementaryPlaneFix) {
    this.seed = seed;
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
