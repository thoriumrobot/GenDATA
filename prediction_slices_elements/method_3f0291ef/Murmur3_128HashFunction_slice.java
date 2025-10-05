// Source-based slice around line 59
// Method: <com.google.common.hash.Murmur3_128HashFunction: int bits()>


  // TODO(user): when the shortcuts are implemented, update BloomFilterStrategies
  private final int seed;

  Murmur3_128HashFunction(int seed) {
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
