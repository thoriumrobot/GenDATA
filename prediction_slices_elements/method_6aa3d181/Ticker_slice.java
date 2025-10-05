// Source-based slice around line 36
// Method: <com.google.common.base.Ticker: long read()>

 * @since 10.0 (<a href="https://github.com/google/guava/wiki/Compatibility">mostly
 *     source-compatible</a> since 9.0)
 */
@GwtCompatible
public abstract class Ticker {
  /** Constructor for use by subclasses. */
  protected Ticker() {}

  /** Returns the number of nanoseconds elapsed since this ticker's fixed point of reference. */
  public abstract long read();

  /**
   * A ticker that reads the current time using {@link System#nanoTime}.
   *
   * @since 10.0
   */
  public static Ticker systemTicker() {
    return SYSTEM_TICKER;
  }

