// Source-based slice around line 31
// Method: <com.google.common.escape.Platform: char[] charBufferFromThreadLocal()>

 * Methods factored out so that they can be emulated differently in GWT.
 *
 * @author Jesse Wilson
 */
@GwtCompatible
final class Platform {
  private Platform() {}

  /** Returns a thread-local 1024-char array. */
  static char[] charBufferFromThreadLocal() {
    // requireNonNull accommodates Android's @RecentlyNullable annotation on ThreadLocal.get
    return requireNonNull(DEST_TL.get());
  }

  /**
   * A thread-local destination buffer to keep us from creating new buffers. The starting size is
   * 1024 characters. If we grow past this we don't put it back in the threadlocal, we just keep
   * going and grow as needed.
   */
  private static final ThreadLocal<char[]> DEST_TL =
