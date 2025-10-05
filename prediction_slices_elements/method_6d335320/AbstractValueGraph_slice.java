// Source-based slice around line 47
// Method: <com.google.common.graph.AbstractValueGraph: Graph asGraph()>

 * @since 20.0
 */
@Beta
public abstract class AbstractValueGraph<N, V> extends AbstractBaseGraph<N>
    implements ValueGraph<N, V> {
  /** Constructor for use by subclasses. */
  public AbstractValueGraph() {}

  @Override
  public Graph<N> asGraph() {
    return new AbstractGraph<N>() {
      @Override
      public Set<N> nodes() {
        return AbstractValueGraph.this.nodes();
      }

      @Override
      public Set<EndpointPair<N>> edges() {
        return AbstractValueGraph.this.edges();
      }
