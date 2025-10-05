// Source-based slice around line 352
// Method: <com.google.common.hash.HashCode: int decode(char)>

    byte[] bytes = new byte[string.length() / 2];
    for (int i = 0; i < string.length(); i += 2) {
      int ch1 = decode(string.charAt(i)) << 4;
      int ch2 = decode(string.charAt(i + 1));
      bytes[i / 2] = (byte) (ch1 + ch2);
    }
    return fromBytesNoCopy(bytes);
  }

  private static int decode(char ch) {
    if (ch >= '0' && ch <= '9') {
      return ch - '0';
    }
    if (ch >= 'a' && ch <= 'f') {
      return ch - 'a' + 10;
    }
    throw new IllegalArgumentException("Illegal hexadecimal character: " + ch);
  }

  /**
