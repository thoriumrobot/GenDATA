// Source-based slice around line 50
// Method: <com.google.common.hash.FarmHashFingerprint64: HashCode hashBytes(byte[],int,int)>

final class FarmHashFingerprint64 extends AbstractNonStreamingHashFunction {
  static final HashFunction FARMHASH_FINGERPRINT_64 = new FarmHashFingerprint64();

  // Some primes between 2^63 and 2^64 for various uses.
  private static final long K0 = 0xc3a5c85c97cb3127L;
  private static final long K1 = 0xb492b66fbe98f273L;
  private static final long K2 = 0x9ae16a3b2f90404fL;

  @Override
  public HashCode hashBytes(byte[] input, int off, int len) {
    checkPositionIndexes(off, off + len, input.length);
    return HashCode.fromLong(fingerprint(input, off, len));
  }

  @Override
  public int bits() {
    return 64;
  }

  @Override
