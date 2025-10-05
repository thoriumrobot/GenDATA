// Source-based slice around line 62
// Method: <com.google.common.graph.StandardMutableValueGraph: boolean addNode(N)>

  }

  @Override
  public ElementOrder<N> incidentEdgeOrder() {
    return incidentEdgeOrder;
  }

  @Override
  @CanIgnoreReturnValue
  public boolean addNode(N node) {
    checkNotNull(node, "node");

    if (containsNode(node)) {
      return false;
    }

    addNodeInternal(node);
    return true;
  }

