// Source-based slice around line 58
// Method: <com.google.common.graph.ImmutableGraph: ImmutableGraph copyOf(Graph)>

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
            new StandardValueGraph<N, Presence>(
                GraphBuilder.from(graph), getNodeConnections(graph), graph.edges().size()));
  }

  /**
   * Simply returns its argument.
   *
