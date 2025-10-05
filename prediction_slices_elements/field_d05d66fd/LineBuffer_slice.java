// Source-based slice around line 37
// Method: com.google.common.io.LineBuffer.line

 * call {@link #finish} at the end of stream.
 *
 * @author Chris Nokleberg
 * @since 1.0
 */
@J2ktIncompatible
@GwtIncompatible
abstract class LineBuffer {
  /** Holds partial line contents. */
  private StringBuilder line = new StringBuilder();

  /** Whether a line ending with a CR is pending processing. */
  private boolean sawReturn;

  /**
   * Process additional characters from the stream. When a line separator is found the contents of
   * the line and the line separator itself are passed to the abstract {@link #handleLine} method.
   *
   * @param cbuf the character buffer to process
   * @param off the offset into the buffer
