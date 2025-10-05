// Source-based slice around line 40
// Method: <com.google.common.graph.StandardMutableGraph: BaseGraph delegate()>

final class StandardMutableGraph<N> extends ForwardingGraph<N> implements MutableGraph<N> {
  private final MutableValueGraph<N, Presence> backingValueGraph;

  /** Constructs a {@link MutableGraph} with the properties specified in {@code builder}. */
  StandardMutableGraph(AbstractGraphBuilder<? super N> builder) {
    this.backingValueGraph = new StandardMutableValueGraph<>(builder);
  }

  @Override
  BaseGraph<N> delegate() {
    return backingValueGraph;
  }

  @Override
  public boolean addNode(N node) {
    return backingValueGraph.addNode(node);
  }

  @Override
  public boolean putEdge(N nodeU, N nodeV) {
