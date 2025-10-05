// Source-based slice around line 187
// Method: <com.google.common.graph.StandardNetwork: N checkedReferenceNode(E)>

  final NetworkConnections<N, E> checkedConnections(N node) {
    NetworkConnections<N, E> connections = nodeConnections.get(node);
    if (connections == null) {
      checkNotNull(node);
      throw new IllegalArgumentException(String.format(NODE_NOT_IN_GRAPH, node));
    }
    return connections;
  }

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
