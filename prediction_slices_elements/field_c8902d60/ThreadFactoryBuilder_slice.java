// Source-based slice around line 60
// Method: com.google.common.util.concurrent.ThreadFactoryBuilder.backingThreadFactory

 * @since 4.0
 */
@J2ktIncompatible
@GwtIncompatible
public final class ThreadFactoryBuilder {
  private @Nullable String nameFormat = null;
  private @Nullable Boolean daemon = null;
  private @Nullable Integer priority = null;
  private @Nullable UncaughtExceptionHandler uncaughtExceptionHandler = null;
  private @Nullable ThreadFactory backingThreadFactory = null;

  /**
   * Creates a new {@link ThreadFactory} builder.
   *
   * <p><b>Java 21+ users:</b> use {@link Thread#ofPlatform()} instead, translating other calls on
   * the builder as documented on each method (except for the rarely used {@link #setThreadFactory},
   * which does not have an equivalent).
   */
  public ThreadFactoryBuilder() {}

