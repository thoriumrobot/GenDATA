// Source-based slice around line 36
// Method: <com.google.common.hash.Crc32cHashFunction: Hasher newHasher()>

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
    return "Hashing.crc32c()";
  }

  static final class Crc32cHasher extends AbstractStreamingHasher {

