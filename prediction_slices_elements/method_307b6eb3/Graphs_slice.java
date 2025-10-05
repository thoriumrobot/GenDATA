// Source-based slice around line 254
// Method: <com.google.common.graph.Graphs: Graph transpose(Graph)>


  // Graph mutation methods

  // Graph view methods

  /**
   * Returns a view of {@code graph} with the direction (if any) of every edge reversed. All other
   * properties remain intact, and further updates to {@code graph} will be reflected in the view.
   */
  public static <N> Graph<N> transpose(Graph<N> graph) {
    if (!graph.isDirected()) {
      return graph; // the transpose of an undirected graph is an identical graph
    }

    if (graph instanceof TransposedGraph) {
      return ((TransposedGraph<N>) graph).graph;
    }

    return new TransposedGraph<>(graph);
  }
