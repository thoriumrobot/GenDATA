// Source-based slice around line 136
// Method: <com.google.common.graph.AbstractDirectedNetworkConnections: void addOutEdge(E,N)>


    if (isSelfLoop) {
      checkPositive(++selfLoopCount);
    }
    N previousNode = inEdgeMap.put(edge, node);
    checkState(previousNode == null);
  }

  @Override
  public void addOutEdge(E edge, N node) {
    checkNotNull(edge);
    checkNotNull(node);

    N previousNode = outEdgeMap.put(edge, node);
    checkState(previousNode == null);
  }
}
