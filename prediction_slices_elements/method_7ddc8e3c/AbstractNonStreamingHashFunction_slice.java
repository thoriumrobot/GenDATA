// Source-based slice around line 66
// Method: <com.google.common.hash.AbstractNonStreamingHashFunction: HashCode hashString(CharSequence,Charset)>

    int len = input.length();
    ByteBuffer buffer = ByteBuffer.allocate(len * 2).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < len; i++) {
      buffer.putChar(input.charAt(i));
    }
    return hashBytes(buffer.array());
  }

  @Override
  public HashCode hashString(CharSequence input, Charset charset) {
    return hashBytes(input.toString().getBytes(charset));
  }

  @Override
  public abstract HashCode hashBytes(byte[] input, int off, int len);

  @Override
  public HashCode hashBytes(ByteBuffer input) {
    return newHasher(input.remaining()).putBytes(input).hash();
  }
