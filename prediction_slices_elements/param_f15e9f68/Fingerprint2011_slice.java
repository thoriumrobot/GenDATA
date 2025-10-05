// Source-based slice around line 59
// Method: <com.google.common.hash.Fingerprint2011: long fingerprint(byte[],int,int)>


  @Override
  public String toString() {
    return "Hashing.fingerprint2011()";
  }

  // End of public functions.

  @VisibleForTesting
  static long fingerprint(byte[] bytes, int offset, int length) {
    long result;

    if (length <= 32) {
      result = murmurHash64WithSeed(bytes, offset, length, K0 ^ K1 ^ K2);
    } else if (length <= 64) {
      result = hashLength33To64(bytes, offset, length);
    } else {
      result = fullFingerprint(bytes, offset, length);
    }

