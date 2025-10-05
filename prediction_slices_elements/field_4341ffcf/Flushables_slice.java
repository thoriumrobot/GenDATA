// Source-based slice around line 34
// Method: com.google.common.io.Flushables.logger

/**
 * Utility methods for working with {@link Flushable} objects.
 *
 * @author Michael Lancaster
 * @since 1.0
 */
@J2ktIncompatible
@GwtIncompatible
public final class Flushables {
  private static final Logger logger = Logger.getLogger(Flushables.class.getName());

  private Flushables() {}

  /**
   * Flush a {@link Flushable}, with control over whether an {@code IOException} may be thrown.
   *
   * <p>If {@code swallowIOException} is true, then we don't rethrow {@code IOException}, but merely
   * log it.
   *
   * @param flushable the {@code Flushable} object to be flushed.
