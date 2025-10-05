// Source-based slice around line 53
// Method: <com.google.common.graph.StandardMutableNetwork: boolean addNode(N)>

    implements MutableNetwork<N, E> {

  /** Constructs a mutable graph with the properties specified in {@code builder}. */
  StandardMutableNetwork(NetworkBuilder<? super N, ? super E> builder) {
    super(builder);
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

