// Source-based slice around line 37
// Method: com.google.common.io.MultiInputStream.it

 * a time.
 *
 * @author Chris Nokleberg
 * @since 1.0
 */
@J2ktIncompatible
@GwtIncompatible
final class MultiInputStream extends InputStream {

  private final Iterator<? extends ByteSource> it;
  private @Nullable InputStream in;

  /**
   * Creates a new instance.
   *
   * @param it an iterator of I/O suppliers that will provide each substream
   */
  public MultiInputStream(Iterator<? extends ByteSource> it) throws IOException {
    this.it = checkNotNull(it);
    advance();
