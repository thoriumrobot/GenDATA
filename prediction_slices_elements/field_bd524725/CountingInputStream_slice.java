// Source-based slice around line 36
// Method: com.google.common.io.CountingInputStream.mark

 *
 * @author Chris Nokleberg
 * @since 1.0
 */
@J2ktIncompatible
@GwtIncompatible
public final class CountingInputStream extends FilterInputStream {

  private long count;
  private long mark = -1;

  /**
   * Wraps another input stream, counting the number of bytes read.
   *
   * @param in the input stream to be wrapped
   */
  public CountingInputStream(InputStream in) {
    super(checkNotNull(in));
  }

