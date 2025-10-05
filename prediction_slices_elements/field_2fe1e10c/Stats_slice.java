// Source-based slice around line 69
// Method: com.google.common.math.Stats.count

 *
 * @author Pete Gillin
 * @author Kevin Bourrillion
 * @since 20.0
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
