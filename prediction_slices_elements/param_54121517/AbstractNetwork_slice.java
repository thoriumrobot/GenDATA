// Source-based slice around line 145
// Method: <com.google.common.graph.AbstractNetwork: int degree(N)>

      public Set<N> successors(N node) {
        return AbstractNetwork.this.successors(node);
      }

      // DO NOT override the AbstractGraph *degree() implementations.
    };
  }

  @Override
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
