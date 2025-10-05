// Source-based slice around line 148
// Method: <com.google.common.graph.AbstractBaseGraph: int inDegree(N)>

      return IntMath.saturatedAdd(predecessors(node).size(), successors(node).size());
    } else {
      Set<N> neighbors = adjacentNodes(node);
      int selfLoopCount = (allowsSelfLoops() && neighbors.contains(node)) ? 1 : 0;
      return IntMath.saturatedAdd(neighbors.size(), selfLoopCount);
    }
  }

  @Override
  public int inDegree(N node) {
    return isDirected() ? predecessors(node).size() : degree(node);
  }

  @Override
  public int outDegree(N node) {
    return isDirected() ? successors(node).size() : degree(node);
  }

  @Override
  public boolean hasEdgeConnecting(N nodeU, N nodeV) {
