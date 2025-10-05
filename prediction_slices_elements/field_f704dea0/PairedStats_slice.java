// Source-based slice around line 44
// Method: com.google.common.math.PairedStats.xStats

 * values (e.g. points on a plane). Build instances with {@link PairedStatsAccumulator#snapshot}.
 *
 * @author Pete Gillin
 * @since 20.0
 */
@J2ktIncompatible
@GwtIncompatible
public final class PairedStats implements Serializable {

  private final Stats xStats;
  private final Stats yStats;
  private final double sumOfProductsOfDeltas;

  /**
   * Internal constructor. Users should use {@link PairedStatsAccumulator#snapshot}.
   *
   * <p>To ensure that the created instance obeys its contract, the parameters should satisfy the
   * following constraints. This is the callers responsibility and is not enforced here.
   *
   * <ul>
