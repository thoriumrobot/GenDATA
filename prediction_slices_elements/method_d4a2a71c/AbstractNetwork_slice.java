// Source-based slice around line 60
// Method: <com.google.common.graph.AbstractNetwork: Graph asGraph()>

 * @param <E> Edge parameter type
 * @since 20.0
 */
@Beta
public abstract class AbstractNetwork<N, E> implements Network<N, E> {
  /** Constructor for use by subclasses. */
  public AbstractNetwork() {}

  @Override
  public Graph<N> asGraph() {
    return new AbstractGraph<N>() {
      @Override
      public Set<N> nodes() {
        return AbstractNetwork.this.nodes();
      }

      @Override
      public Set<EndpointPair<N>> edges() {
        if (allowsParallelEdges()) {
          return super.edges(); // Defer to AbstractGraph implementation.
