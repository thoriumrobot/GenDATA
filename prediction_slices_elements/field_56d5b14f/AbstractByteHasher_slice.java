// Source-based slice around line 36
// Method: com.google.common.hash.AbstractByteHasher.scratch

import org.jspecify.annotations.Nullable;

/**
 * Abstract {@link Hasher} that handles converting primitives to bytes using a scratch {@code
 * ByteBuffer} and streams all bytes to a sink to compute the hash.
 *
 * @author Colin Decker
 */
abstract class AbstractByteHasher extends AbstractHasher {
  private @Nullable ByteBuffer scratch;

  /** Updates this hasher with the given byte. */
  protected abstract void update(byte b);

  /** Updates this hasher with the given bytes. */
  protected void update(byte[] b) {
    update(b, 0, b.length);
  }

  /** Updates this hasher with {@code len} bytes starting at {@code off} in the given buffer. */
