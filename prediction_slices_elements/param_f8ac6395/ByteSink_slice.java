// Source-based slice around line 60
// Method: <com.google.common.io.ByteSink: CharSink asCharSink(Charset)>

public abstract class ByteSink {

  /** Constructor for use by subclasses. */
  protected ByteSink() {}

  /**
   * Returns a {@link CharSink} view of this {@code ByteSink} that writes characters to this sink as
   * bytes encoded with the given {@link Charset charset}.
   */
  public CharSink asCharSink(Charset charset) {
    return new AsCharSink(charset);
  }

  /**
   * Opens a new {@link OutputStream} for writing to this sink. This method returns a new,
   * independent stream each time it is called.
   *
   * <p>The caller is responsible for ensuring that the returned stream is closed.
   *
   * @throws IOException if an I/O error occurs while opening the stream
