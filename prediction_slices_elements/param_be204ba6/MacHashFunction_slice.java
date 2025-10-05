// Source-based slice around line 67
// Method: <com.google.common.hash.MacHashFunction: Mac getMac(String,Key)>

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
      Mac mac = Mac.getInstance(algorithmName);
      mac.init(key);
      return mac;
    } catch (NoSuchAlgorithmException e) {
      throw new IllegalStateException(e);
    } catch (InvalidKeyException e) {
      throw new IllegalArgumentException(e);
    }
  }
