// Source-based slice around line 152
// Method: <com.google.common.hash.Murmur3_32HashFunction: HashCode hashString(CharSequence,Charset)>

      int k1 = input.charAt(input.length() - 1);
      k1 = mixK1(k1);
      h1 ^= k1;
    }

    return fmix(h1, Chars.BYTES * input.length());
  }

  @Override
  public HashCode hashString(CharSequence input, Charset charset) {
    if (charset.equals(UTF_8)) {
      int utf16Length = input.length();
      int h1 = seed;
      int i = 0;
      int len = 0;

      // This loop optimizes for pure ASCII.
      while (i + 4 <= utf16Length) {
        char c0 = input.charAt(i);
        char c1 = input.charAt(i + 1);
