// Source-based slice around line 37
// Method: com.google.common.io.Closeables.logger

/**
 * Utility methods for working with {@link Closeable} objects.
 *
 * @author Michael Lancaster
 * @since 1.0
 */
@J2ktIncompatible
@GwtIncompatible
public final class Closeables {
  @VisibleForTesting static final Logger logger = Logger.getLogger(Closeables.class.getName());

  private Closeables() {}

  /**
   * Closes a {@link Closeable}, with control over whether an {@code IOException} may be thrown.
   * This is primarily useful in a finally block, where a thrown exception needs to be logged but
   * not propagated (otherwise the original exception will be lost).
   *
   * <p>If {@code swallowIOException} is true then we never throw {@code IOException} but merely log
   * it.
