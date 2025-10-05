// Source-based slice around line 84
// Method: <com.google.common.graph.StandardValueGraph: Set nodes()>

    // Prefer the heavier "MapRetrievalCache" for nodes if lookup is expensive.
    this.nodeConnections =
        (nodeConnections instanceof TreeMap)
            ? new MapRetrievalCache<N, GraphConnections<N, V>>(nodeConnections)
            : new MapIteratorCache<N, GraphConnections<N, V>>(nodeConnections);
    this.edgeCount = checkNonNegative(edgeCount);
  }

  @Override
  public Set<N> nodes() {
    return nodeConnections.unmodifiableKeySet();
  }

  @Override
  public boolean isDirected() {
    return isDirected;
  }

  @Override
  public boolean allowsSelfLoops() {
