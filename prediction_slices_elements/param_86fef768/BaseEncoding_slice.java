// Source-based slice around line 244
// Method: <com.google.common.io.BaseEncoding: InputStream decodingStream(Reader)>

    return extract(tmp, len);
  }

  /**
   * Returns an {@code InputStream} that decodes base-encoded input from the specified {@code
   * Reader}. The returned stream throws a {@link DecodingException} upon decoding-specific errors.
   */
  @J2ktIncompatible
  @GwtIncompatible // Reader,InputStream
  public abstract InputStream decodingStream(Reader reader);

  /**
   * Returns a {@code ByteSource} that reads base-encoded bytes from the specified {@code
   * CharSource}.
   */
  @J2ktIncompatible
  @GwtIncompatible // ByteSource,CharSource
  public final ByteSource decodingSource(CharSource encodedSource) {
    checkNotNull(encodedSource);
    return new ByteSource() {
