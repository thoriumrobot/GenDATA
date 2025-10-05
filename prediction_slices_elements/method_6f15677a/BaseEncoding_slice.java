// Source-based slice around line 172
// Method: <com.google.common.io.BaseEncoding: OutputStream encodingStream(Writer)>

  }

  /**
   * Returns an {@code OutputStream} that encodes bytes using this encoding into the specified
   * {@code Writer}. When the returned {@code OutputStream} is closed, so is the backing {@code
   * Writer}.
   */
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
