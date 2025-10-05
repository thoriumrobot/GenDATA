// Source-based slice around line 73
// Method: com.google.common.math.Stats.max

 */
@J2ktIncompatible
@GwtIncompatible
public final class Stats implements Serializable {

  private final long count;
  private final double mean;
  private final double sumOfSquaresOfDeltas;
  private final double min;
  private final double max;

  /**
   * Internal constructor. Users should use {@link #of} or {@link StatsAccumulator#snapshot}.
   *
   * <p>To ensure that the created instance obeys its contract, the parameters should satisfy the
   * following constraints. This is the callers responsibility and is not enforced here.
   *
   * <ul>
   *   <li>If {@code count} is 0, {@code mean} may have any finite value (its only usage will be to
   *       get multiplied by 0 to calculate the sum), and the other parameters may have any values
