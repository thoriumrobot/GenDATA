// Source-based slice around line 107
// Method: <com.google.common.graph.ImmutableGraph: BaseGraph delegate()>

    Function<N, Presence> edgeValueFn =
        (Function<N, Presence>) Functions.constant(Presence.EDGE_EXISTS);
    return graph.isDirected()
        ? DirectedGraphConnections.ofImmutable(node, graph.incidentEdges(node), edgeValueFn)
        : UndirectedGraphConnections.ofImmutable(
            Maps.asMap(graph.adjacentNodes(node), edgeValueFn));
  }

  @Override
  BaseGraph<N> delegate() {
    return backingGraph;
  }

  /**
   * A builder for creating {@link ImmutableGraph} instances, especially {@code static final}
   * graphs. Example:
   *
   * {@snippet :
   * static final ImmutableGraph<Country> COUNTRY_ADJACENCY_GRAPH =
   *     GraphBuilder.undirected()
