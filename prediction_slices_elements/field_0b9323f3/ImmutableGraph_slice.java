// Source-based slice around line 51
// Method: com.google.common.graph.ImmutableGraph.backingGraph

 * @author Omar Darwish
 * @author Jens Nyman
 * @param <N> Node parameter type
 * @since 20.0
 */
@Beta
@Immutable(containerOf = {"N"})
public class ImmutableGraph<N> extends ForwardingGraph<N> {
  @SuppressWarnings("Immutable") // The backing graph must be immutable.
  private final BaseGraph<N> backingGraph;

  ImmutableGraph(BaseGraph<N> backingGraph) {
    this.backingGraph = backingGraph;
  }

  /** Returns an immutable copy of {@code graph}. */
  public static <N> ImmutableGraph<N> copyOf(Graph<N> graph) {
    return (graph instanceof ImmutableGraph)
        ? (ImmutableGraph<N>) graph
        : new ImmutableGraph<N>(
