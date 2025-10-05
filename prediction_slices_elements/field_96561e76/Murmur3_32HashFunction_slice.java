// Source-based slice around line 72
// Method: com.google.common.hash.Murmur3_32HashFunction.supplementaryPlaneFix

  static final HashFunction GOOD_FAST_HASH_32 =
      new Murmur3_32HashFunction(Hashing.GOOD_FAST_HASH_SEED, /* supplementaryPlaneFix= */ true);

  private static final int CHUNK_SIZE = 4;

  private static final int C1 = 0xcc9e2d51;
  private static final int C2 = 0x1b873593;

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
