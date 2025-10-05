// Source-based slice around line 52
// Method: <com.google.common.hash.Fingerprint2011: String toString()>

    return HashCode.fromLong(fingerprint(input, off, len));
  }

  @Override
  public int bits() {
    return 64;
  }

  @Override
  public String toString() {
    return "Hashing.fingerprint2011()";
  }

  // End of public functions.

  @VisibleForTesting
  static long fingerprint(byte[] bytes, int offset, int length) {
    long result;

    if (length <= 32) {
