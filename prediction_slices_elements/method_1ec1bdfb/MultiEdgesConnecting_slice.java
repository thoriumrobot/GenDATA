// Source-based slice around line 49
// Method: <com.google.common.graph.MultiEdgesConnecting: UnmodifiableIterator iterator()>

  private final Map<E, ?> outEdgeToNode;
  private final Object targetNode;

  MultiEdgesConnecting(Map<E, ?> outEdgeToNode, Object targetNode) {
    this.outEdgeToNode = checkNotNull(outEdgeToNode);
    this.targetNode = checkNotNull(targetNode);
  }

  @Override
  public UnmodifiableIterator<E> iterator() {
    Iterator<? extends Entry<E, ?>> entries = outEdgeToNode.entrySet().iterator();
    return new AbstractIterator<E>() {
      @Override
      protected @Nullable E computeNext() {
        while (entries.hasNext()) {
          Entry<E, ?> entry = entries.next();
          if (targetNode.equals(entry.getValue())) {
            return entry.getKey();
          }
        }
