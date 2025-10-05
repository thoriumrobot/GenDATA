// Source-based slice around line 58
// Method: <com.google.common.hash.MacHashFunction: boolean supportsClone(Mac)>

    this.bits = prototype.getMacLength() * Byte.SIZE;
    this.supportsClone = supportsClone(prototype);
  }

  @Override
  public int bits() {
    return bits;
  }

  private static boolean supportsClone(Mac mac) {
    try {
      Object unused = mac.clone();
      return true;
    } catch (CloneNotSupportedException e) {
      return false;
    }
  }

  private static Mac getMac(String algorithmName, Key key) {
    try {
