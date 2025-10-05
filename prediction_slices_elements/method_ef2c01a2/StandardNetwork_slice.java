// Source-based slice around line 196
// Method: <com.google.common.graph.StandardNetwork: boolean containsNode(N)>

  final N checkedReferenceNode(E edge) {
    N referenceNode = edgeToReferenceNode.get(edge);
    if (referenceNode == null) {
      checkNotNull(edge);
      throw new IllegalArgumentException(String.format(EDGE_NOT_IN_GRAPH, edge));
    }
    return referenceNode;
  }

  final boolean containsNode(N node) {
    return nodeConnections.containsKey(node);
  }

  final boolean containsEdge(E edge) {
    return edgeToReferenceNode.containsKey(edge);
  }
}
