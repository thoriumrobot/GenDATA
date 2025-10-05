// Source-based slice around line 48
// Method: com.google.common.hash.Murmur3_128HashFunction.GOOD_FAST_HASH_128

 *
 * @author Austin Appleby
 * @author Dimitris Andreou
 */
@Immutable
@SuppressWarnings("IdentifierName") // the best we could do for adjacent digit blocks
final class Murmur3_128HashFunction extends AbstractHashFunction implements Serializable {
  static final HashFunction MURMUR3_128 = new Murmur3_128HashFunction(0);

  static final HashFunction GOOD_FAST_HASH_128 =
      new Murmur3_128HashFunction(Hashing.GOOD_FAST_HASH_SEED);

  // TODO(user): when the shortcuts are implemented, update BloomFilterStrategies
  private final int seed;

  Murmur3_128HashFunction(int seed) {
    this.seed = seed;
  }

  @Override
