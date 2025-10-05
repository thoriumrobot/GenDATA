// Source-based slice around line 252
// Method: <com.google.common.io.BaseEncoding: ByteSource decodingSource(CharSource)>

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
      @Override
      public InputStream openStream() throws IOException {
        return decodingStream(encodedSource.openStream());
      }
    };
  }

  // Implementations for encoding/decoding
