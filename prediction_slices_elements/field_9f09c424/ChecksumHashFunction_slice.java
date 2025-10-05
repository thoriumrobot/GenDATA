// Source-based slice around line 38
// Method: com.google.common.hash.ChecksumHashFunction.checksumSupplier

import org.jspecify.annotations.Nullable;

/**
 * {@link HashFunction} adapter for {@link Checksum} instances.
 *
 * @author Colin Decker
 */
@Immutable
final class ChecksumHashFunction extends AbstractHashFunction implements Serializable {
  private final ImmutableSupplier<? extends Checksum> checksumSupplier;
  private final int bits;
  private final String toString;

  ChecksumHashFunction(
      ImmutableSupplier<? extends Checksum> checksumSupplier, int bits, String toString) {
    this.checksumSupplier = checkNotNull(checksumSupplier);
    checkArgument(bits == 32 || bits == 64, "bits (%s) must be either 32 or 64", bits);
    this.bits = bits;
    this.toString = checkNotNull(toString);
  }
