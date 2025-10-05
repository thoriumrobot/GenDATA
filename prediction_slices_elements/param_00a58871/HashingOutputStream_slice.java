// Source-based slice around line 56
// Method: <com.google.common.hash.HashingOutputStream: void write(byte[],int,int)>

  }

  @Override
  public void write(int b) throws IOException {
    hasher.putByte((byte) b);
    out.write(b);
  }

  @Override
  public void write(byte[] bytes, int off, int len) throws IOException {
    hasher.putBytes(bytes, off, len);
    out.write(bytes, off, len);
  }

  /**
   * Returns the {@link HashCode} based on the data written to this stream. The result is
   * unspecified if this method is called more than once on the same instance.
   */
  public HashCode hash() {
    return hasher.hash();
