// Source-based slice around line 42
// Method: <com.google.common.graph.IncidentEdgeSet: int size()>

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
      return graph.adjacentNodes(node).size();
    }
  }

  @Override
