// Source-based slice around line 69
// Method: com.google.common.hash.Murmur3_32HashFunction.C2


  // We can include the non-BMP fix here because Hashing.goodFastHash stresses that the hash is a
  // temporary-use one. Therefore it shouldn't be persisted.
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
