// Source-based slice around line 31
// Method: <com.google.common.hash.Crc32cHashFunction: int bits()>

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

  @Override
  public String toString() {
