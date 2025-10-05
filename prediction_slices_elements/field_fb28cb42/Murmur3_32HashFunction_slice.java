// Source-based slice around line 56
// Method: com.google.common.hash.Murmur3_32HashFunction.MURMUR3_32

 * implementation</a>.
 *
 * @author Austin Appleby
 * @author Dimitris Andreou
 * @author Kurt Alfred Kluever
 */
@Immutable
@SuppressWarnings("IdentifierName") // the best we could do for adjacent digit blocks
final class Murmur3_32HashFunction extends AbstractHashFunction implements Serializable {
  static final HashFunction MURMUR3_32 =
      new Murmur3_32HashFunction(0, /* supplementaryPlaneFix= */ false);
  static final HashFunction MURMUR3_32_FIXED =
      new Murmur3_32HashFunction(0, /* supplementaryPlaneFix= */ true);

  // We can include the non-BMP fix here because Hashing.goodFastHash stresses that the hash is a
  // temporary-use one. Therefore it shouldn't be persisted.
  static final HashFunction GOOD_FAST_HASH_32 =
      new Murmur3_32HashFunction(Hashing.GOOD_FAST_HASH_SEED, /* supplementaryPlaneFix= */ true);

  private static final int CHUNK_SIZE = 4;
