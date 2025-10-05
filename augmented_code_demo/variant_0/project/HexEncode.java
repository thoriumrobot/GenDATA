    @Positive
public class HexEncode {
    @Positive
  private static final char[] digits = {
    @Positive
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'
    @Positive
  };

    @Positive
  public static String hexEncode(byte[] bytes) {
    @Positive
    StringBuffer s = new StringBuffer(bytes.length * 2);
    @Positive
    for (int i = 0; i < bytes.length; i++) {
    @Positive
      byte b = bytes[i];
    @Positive
      s.append(digits[(b & 0xf0) >> 4]);
    @Positive
      s.append(digits[b & 0x0f]);
    @Positive
    }
    @Positive
    return s.toString();
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
