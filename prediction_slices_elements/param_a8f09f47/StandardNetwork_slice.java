// Source-based slice around line 200
// Method: <com.google.common.graph.StandardNetwork: boolean containsEdge(E)>

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
