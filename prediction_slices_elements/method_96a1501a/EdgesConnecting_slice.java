// Source-based slice around line 61
// Method: <com.google.common.graph.EdgesConnecting: boolean contains(Object)>

        : Iterators.singletonIterator(connectingEdge);
  }

  @Override
  public int size() {
    return getConnectingEdge() == null ? 0 : 1;
  }

  @Override
  public boolean contains(@Nullable Object edge) {
    E connectingEdge = getConnectingEdge();
    return connectingEdge != null && connectingEdge.equals(edge);
  }

  private @Nullable E getConnectingEdge() {
    return nodeToOutEdge.get(targetNode);
  }
}
