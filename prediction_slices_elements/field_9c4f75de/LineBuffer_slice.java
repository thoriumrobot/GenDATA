// Source-based slice around line 40
// Method: com.google.common.io.LineBuffer.sawReturn

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
   * @param len the number of characters to process
   * @throws IOException if an I/O error occurs
   * @see #finish
