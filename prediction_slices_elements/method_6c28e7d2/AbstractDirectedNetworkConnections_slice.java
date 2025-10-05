// Source-based slice around line 94
// Method: <com.google.common.graph.AbstractDirectedNetworkConnections: Set outEdges()>

    };
  }

  @Override
  public Set<E> inEdges() {
    return Collections.unmodifiableSet(inEdgeMap.keySet());
  }

  @Override
  public Set<E> outEdges() {
    return Collections.unmodifiableSet(outEdgeMap.keySet());
  }

  @Override
  public N adjacentNode(E edge) {
    // Since the reference node is defined to be 'source' for directed graphs,
    // we can assume this edge lives in the set of outgoing edges.
    // (We're relying on callers to call this method only with an edge that's in the graph.)
    return requireNonNull(outEdgeMap.get(edge));
  }
