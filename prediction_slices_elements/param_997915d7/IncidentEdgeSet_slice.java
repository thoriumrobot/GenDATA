// Source-based slice around line 53
// Method: <com.google.common.graph.IncidentEdgeSet: boolean contains(Object)>

      return graph.inDegree(node)
          + graph.outDegree(node)
          - (graph.successors(node).contains(node) ? 1 : 0);
    } else {
      return graph.adjacentNodes(node).size();
    }
  }

  @Override
  public boolean contains(@Nullable Object obj) {
    if (!(obj instanceof EndpointPair)) {
      return false;
    }
    EndpointPair<?> endpointPair = (EndpointPair<?>) obj;

    if (graph.isDirected()) {
      if (!endpointPair.isOrdered()) {
        return false;
      }

