// Source-based slice around line 50
// Method: <com.google.common.hash.HashingOutputStream: void write(int)>

  // HashCode on an existing OutputStream, compared to creating a separate OutputStream that could
  // be (optionally) be combined with another if needed (with something like
  // MultiplexingOutputStream).
  public HashingOutputStream(HashFunction hashFunction, OutputStream out) {
    super(checkNotNull(out));
    this.hasher = checkNotNull(hashFunction.newHasher());
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

