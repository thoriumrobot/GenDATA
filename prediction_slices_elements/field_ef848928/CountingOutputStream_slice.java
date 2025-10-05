// Source-based slice around line 35
// Method: com.google.common.io.CountingOutputStream.count

 * An OutputStream that counts the number of bytes written.
 *
 * @author Chris Nokleberg
 * @since 1.0
 */
@J2ktIncompatible
@GwtIncompatible
public final class CountingOutputStream extends FilterOutputStream {

  private long count;

  /**
   * Wraps another output stream, counting the number of bytes written.
   *
   * @param out the output stream to be wrapped
   */
  public CountingOutputStream(OutputStream out) {
    super(checkNotNull(out));
  }

