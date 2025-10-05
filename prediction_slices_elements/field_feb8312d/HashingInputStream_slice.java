// Source-based slice around line 33
// Method: com.google.common.hash.HashingInputStream.hasher


/**
 * An {@link InputStream} that maintains a hash of the data read from it.
 *
 * @author Qian Huang
 * @since 16.0
 */
@Beta
public final class HashingInputStream extends FilterInputStream {
  private final Hasher hasher;

  /**
   * Creates an input stream that hashes using the given {@link HashFunction} and delegates all data
   * read from it to the underlying {@link InputStream}.
   *
   * <p>The {@link InputStream} should not be read from before or after the hand-off.
   */
  public HashingInputStream(HashFunction hashFunction, InputStream in) {
    super(checkNotNull(in));
    this.hasher = checkNotNull(hashFunction.newHasher());
