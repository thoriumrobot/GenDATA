// Source-based slice around line 38
// Method: com.google.common.io.AppendableWriter.target

 * or {@link Closeable}, flush()es and close()s will also be delegated to the target.
 *
 * @author Alan Green
 * @author Sebastian Kanthak
 * @since 1.0
 */
@J2ktIncompatible
@GwtIncompatible
final class AppendableWriter extends Writer {
  private final Appendable target;
  private boolean closed;

  /**
   * Creates a new writer that appends everything it writes to {@code target}.
   *
   * @param target target to which to append output
   */
  AppendableWriter(Appendable target) {
    this.target = checkNotNull(target);
  }
