// Source-based slice around line 96
// Method: <com.google.common.graph.StandardNetwork: Set nodes()>

    // methods that access the same node(s) repeatedly, such as Graphs.removeEdgesConnecting().
    this.nodeConnections =
        (nodeConnections instanceof TreeMap)
            ? new MapRetrievalCache<N, NetworkConnections<N, E>>(nodeConnections)
            : new MapIteratorCache<N, NetworkConnections<N, E>>(nodeConnections);
    this.edgeToReferenceNode = new MapIteratorCache<>(edgeToReferenceNode);
  }

  @Override
  public Set<N> nodes() {
    return nodeConnections.unmodifiableKeySet();
  }

  @Override
  public Set<E> edges() {
    return edgeToReferenceNode.unmodifiableKeySet();
  }

  @Override
  public boolean isDirected() {
