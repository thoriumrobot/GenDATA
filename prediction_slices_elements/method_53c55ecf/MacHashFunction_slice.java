// Source-based slice around line 54
// Method: <com.google.common.hash.MacHashFunction: int bits()>

  MacHashFunction(String algorithmName, Key key, String toString) {
    this.prototype = getMac(algorithmName, key);
    this.key = checkNotNull(key);
    this.toString = checkNotNull(toString);
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
