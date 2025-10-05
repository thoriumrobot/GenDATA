// Source-based slice around line 32
// Method: com.google.common.hash.HashingOutputStream.hasher


/**
 * An {@link OutputStream} that maintains a hash of the data written to it.
 *
 * @author Zoe Piepmeier
 * @since 16.0
 */
@Beta
public final class HashingOutputStream extends FilterOutputStream {
  private final Hasher hasher;

  /**
   * Creates an output stream that hashes using the given {@link HashFunction}, and forwards all
   * data written to it to the underlying {@link OutputStream}.
   *
   * <p>The {@link OutputStream} should not be written to before or after the hand-off.
   */
  // TODO(user): Evaluate whether it makes sense to always piggyback the computation of a
  // HashCode on an existing OutputStream, compared to creating a separate OutputStream that could
  // be (optionally) be combined with another if needed (with something like
