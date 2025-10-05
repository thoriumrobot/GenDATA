// Source-based slice around line 68
// Method: <com.google.common.hash.FarmHashFingerprint64: long fingerprint(byte[],int,int)>


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
      } else {
        return hashLength17to32(bytes, offset, length);
      }
    } else if (length <= 64) {
      return hashLength33To64(bytes, offset, length);
    } else {
      return hashLength65Plus(bytes, offset, length);
