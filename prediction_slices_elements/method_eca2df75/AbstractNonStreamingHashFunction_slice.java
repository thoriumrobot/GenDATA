// Source-based slice around line 56
// Method: <com.google.common.hash.AbstractNonStreamingHashFunction: HashCode hashUnencodedChars(CharSequence)>

    return hashBytes(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(input).array());
  }

  @Override
  public HashCode hashLong(long input) {
    return hashBytes(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(input).array());
  }

  @Override
  public HashCode hashUnencodedChars(CharSequence input) {
    int len = input.length();
    ByteBuffer buffer = ByteBuffer.allocate(len * 2).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < len; i++) {
      buffer.putChar(input.charAt(i));
    }
    return hashBytes(buffer.array());
  }

  @Override
  public HashCode hashString(CharSequence input, Charset charset) {
