// Source-based slice around line 28
// Method: com.google.common.hash.Crc32cHashFunction.CRC_32_C


/**
 * This class generates a CRC32C checksum, defined by RFC 3720, Section 12.1. The generator
 * polynomial for this checksum is {@code 0x11EDC6F41}.
 *
 * @author Kurt Alfred Kluever
 */
@Immutable
final class Crc32cHashFunction extends AbstractHashFunction {
  static final HashFunction CRC_32_C = new Crc32cHashFunction();

  @Override
  public int bits() {
    return 32;
  }

  @Override
  public Hasher newHasher() {
    return new Crc32cHasher();
  }
