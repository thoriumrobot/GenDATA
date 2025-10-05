// Source-based slice around line 55
// Method: <com.google.common.graph.ImmutableValueGraph: ImmutableValueGraph copyOf(ValueGraph)>

@Immutable(containerOf = {"N", "V"})
@SuppressWarnings("Immutable") // Extends StandardValueGraph but uses ImmutableMaps.
public final class ImmutableValueGraph<N, V> extends StandardValueGraph<N, V> {

  private ImmutableValueGraph(ValueGraph<N, V> graph) {
    super(ValueGraphBuilder.from(graph), getNodeConnections(graph), graph.edges().size());
  }

  /** Returns an immutable copy of {@code graph}. */
  public static <N, V> ImmutableValueGraph<N, V> copyOf(ValueGraph<N, V> graph) {
    return (graph instanceof ImmutableValueGraph)
        ? (ImmutableValueGraph<N, V>) graph
        : new ImmutableValueGraph<N, V>(graph);
  }

  /**
   * Simply returns its argument.
   *
   * @deprecated no need to use this
   */
