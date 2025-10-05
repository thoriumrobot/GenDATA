// Source-based slice around line 100
// Method: com.google.common.base.Stopwatch.ticker

 *      });
 * }
 *
 * @author Kevin Bourrillion
 * @since 10.0
 */
@GwtCompatible
@SuppressWarnings("GoodTime") // lots of violations
public final class Stopwatch {
  private final Ticker ticker;
  private boolean isRunning;
  private long elapsedNanos;
  private long startTick;

  /**
   * Creates (but does not start) a new stopwatch using {@link System#nanoTime} as its time source.
   *
   * @since 15.0
   */
  public static Stopwatch createUnstarted() {
