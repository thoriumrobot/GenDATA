// Source-based slice around line 61
// Method: <com.google.common.hash.FarmHashFingerprint64: String toString()>

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

  @VisibleForTesting
  static long fingerprint(byte[] bytes, int offset, int length) {
    if (length <= 32) {
      if (length <= 16) {
        return hashLength0to16(bytes, offset, length);
