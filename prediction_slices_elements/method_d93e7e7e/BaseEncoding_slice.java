// Source-based slice around line 179
// Method: <com.google.common.io.BaseEncoding: ByteSink encodingSink(CharSink)>

  @J2ktIncompatible
  @GwtIncompatible // Writer,OutputStream
  public abstract OutputStream encodingStream(Writer writer);

  /**
   * Returns a {@code ByteSink} that writes base-encoded bytes to the specified {@code CharSink}.
   */
  @J2ktIncompatible
  @GwtIncompatible // ByteSink,CharSink
  public final ByteSink encodingSink(CharSink encodedSink) {
    checkNotNull(encodedSink);
    return new ByteSink() {
      @Override
      public OutputStream openStream() throws IOException {
        return encodingStream(encodedSink.openStream());
      }
    };
  }

  // TODO(lowasser): document the extent of leniency, probably after adding ignore(CharMatcher)
