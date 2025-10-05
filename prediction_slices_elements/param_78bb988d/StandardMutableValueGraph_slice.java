// Source-based slice around line 162
// Method: <com.google.common.graph.StandardMutableValueGraph: V removeEdge(N,N)>

      }
    }
    nodeConnections.remove(node);
    checkNonNegative(edgeCount);
    return true;
  }

  @Override
  @CanIgnoreReturnValue
  public @Nullable V removeEdge(N nodeU, N nodeV) {
    checkNotNull(nodeU, "nodeU");
    checkNotNull(nodeV, "nodeV");

    GraphConnections<N, V> connectionsU = nodeConnections.get(nodeU);
    GraphConnections<N, V> connectionsV = nodeConnections.get(nodeV);
    if (connectionsU == null || connectionsV == null) {
      return null;
    }

    V previousValue = connectionsU.removeSuccessor(nodeV);
