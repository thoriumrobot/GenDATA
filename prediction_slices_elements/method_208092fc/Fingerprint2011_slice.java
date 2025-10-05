// Source-based slice around line 47
// Method: <com.google.common.hash.Fingerprint2011: int bits()>

  private static final long K3 = 0xc6a4a7935bd1e995L;

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
    return "Hashing.fingerprint2011()";
  }

  // End of public functions.

