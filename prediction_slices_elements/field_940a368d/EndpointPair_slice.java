// Source-based slice around line 44
// Method: com.google.common.graph.EndpointPair.nodeV

 * <p>The edge is a self-loop if, and only if, the two endpoints are equal.
 *
 * @author James Sexton
 * @since 20.0
 */
@Beta
@Immutable(containerOf = {"N"})
public abstract class EndpointPair<N> implements Iterable<N> {
  private final N nodeU;
  private final N nodeV;

  private EndpointPair(N nodeU, N nodeV) {
    this.nodeU = checkNotNull(nodeU);
    this.nodeV = checkNotNull(nodeV);
  }

  /** Returns an {@link EndpointPair} representing the endpoints of a directed edge. */
  public static <N> EndpointPair<N> ordered(N source, N target) {
    return new Ordered<>(source, target);
  }
