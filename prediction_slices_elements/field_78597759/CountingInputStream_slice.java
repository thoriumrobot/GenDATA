// Source-based slice around line 35
// Method: com.google.common.io.CountingInputStream.count

 * An {@link InputStream} that counts the number of bytes read.
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
