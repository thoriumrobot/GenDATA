// Source-based slice around line 270
// Method: <com.google.common.graph.Graphs: ValueGraph transpose(ValueGraph)>

    }

    return new TransposedGraph<>(graph);
  }

  /**
   * Returns a view of {@code graph} with the direction (if any) of every edge reversed. All other
   * properties remain intact, and further updates to {@code graph} will be reflected in the view.
   */
  public static <N, V> ValueGraph<N, V> transpose(ValueGraph<N, V> graph) {
    if (!graph.isDirected()) {
      return graph; // the transpose of an undirected graph is an identical graph
    }

    if (graph instanceof TransposedValueGraph) {
      return ((TransposedValueGraph<N, V>) graph).graph;
    }

    return new TransposedValueGraph<>(graph);
  }
