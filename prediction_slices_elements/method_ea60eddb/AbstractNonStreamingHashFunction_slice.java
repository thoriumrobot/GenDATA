// Source-based slice around line 35
// Method: <com.google.common.hash.AbstractNonStreamingHashFunction: Hasher newHasher()>

 * Skeleton implementation of {@link HashFunction}, appropriate for non-streaming algorithms. All
 * the hash computation done using {@linkplain #newHasher()} are delegated to the {@linkplain
 * #hashBytes(byte[], int, int)} method.
 *
 * @author Dimitris Andreou
 */
@Immutable
abstract class AbstractNonStreamingHashFunction extends AbstractHashFunction {
  @Override
  public Hasher newHasher() {
    return newHasher(32);
  }

  @Override
  public Hasher newHasher(int expectedInputSize) {
    Preconditions.checkArgument(expectedInputSize >= 0);
    return new BufferingHasher(expectedInputSize);
  }

  @Override
