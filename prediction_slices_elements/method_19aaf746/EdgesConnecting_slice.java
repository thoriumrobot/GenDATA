// Source-based slice around line 48
// Method: <com.google.common.graph.EdgesConnecting: UnmodifiableIterator iterator()>

  private final Map<?, E> nodeToOutEdge;
  private final Object targetNode;

  EdgesConnecting(Map<?, E> nodeToEdgeMap, Object targetNode) {
    this.nodeToOutEdge = checkNotNull(nodeToEdgeMap);
    this.targetNode = checkNotNull(targetNode);
  }

  @Override
  public UnmodifiableIterator<E> iterator() {
    E connectingEdge = getConnectingEdge();
    return (connectingEdge == null)
        ? ImmutableSet.<E>of().iterator()
        : Iterators.singletonIterator(connectingEdge);
  }

  @Override
  public int size() {
    return getConnectingEdge() == null ? 0 : 1;
  }
