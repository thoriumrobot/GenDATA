// Source-based slice around line 37
// Method: <com.google.common.graph.IncidentEdgeSet: boolean remove(Object)>

  final N node;
  final BaseGraph<N> graph;

  IncidentEdgeSet(BaseGraph<N> graph, N node) {
    this.graph = graph;
    this.node = node;
  }

  @Override
  public boolean remove(@Nullable Object o) {
    throw new UnsupportedOperationException();
  }

  @Override
  public int size() {
    if (graph.isDirected()) {
      return graph.inDegree(node)
          + graph.outDegree(node)
          - (graph.successors(node).contains(node) ? 1 : 0);
    } else {
