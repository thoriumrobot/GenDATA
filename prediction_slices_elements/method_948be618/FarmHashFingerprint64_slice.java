// Source-based slice around line 56
// Method: <com.google.common.hash.FarmHashFingerprint64: int bits()>

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
  public String toString() {
    return "Hashing.farmHashFingerprint64()";
  }

  // End of public functions.

