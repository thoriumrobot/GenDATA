// Source-based slice around line 154
// Method: <com.google.common.graph.AbstractNetwork: int inDegree(N)>

  public int degree(N node) {
    if (isDirected()) {
      return IntMath.saturatedAdd(inEdges(node).size(), outEdges(node).size());
    } else {
      return IntMath.saturatedAdd(incidentEdges(node).size(), edgesConnecting(node, node).size());
    }
  }

  @Override
  public int inDegree(N node) {
    return isDirected() ? inEdges(node).size() : degree(node);
  }

  @Override
  public int outDegree(N node) {
    return isDirected() ? outEdges(node).size() : degree(node);
  }

  @Override
  public Set<E> adjacentEdges(E edge) {
