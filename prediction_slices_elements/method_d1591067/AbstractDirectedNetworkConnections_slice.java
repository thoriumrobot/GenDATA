// Source-based slice around line 89
// Method: <com.google.common.graph.AbstractDirectedNetworkConnections: Set inEdges()>


      @Override
      public boolean contains(@Nullable Object obj) {
        return inEdgeMap.containsKey(obj) || outEdgeMap.containsKey(obj);
      }
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
