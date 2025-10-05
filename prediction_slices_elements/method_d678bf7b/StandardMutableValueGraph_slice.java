// Source-based slice around line 56
// Method: <com.google.common.graph.StandardMutableValueGraph: ElementOrder incidentEdgeOrder()>

  private final ElementOrder<N> incidentEdgeOrder;

  /** Constructs a mutable graph with the properties specified in {@code builder}. */
  StandardMutableValueGraph(AbstractGraphBuilder<? super N> builder) {
    super(builder);
    incidentEdgeOrder = builder.incidentEdgeOrder.cast();
  }

  @Override
  public ElementOrder<N> incidentEdgeOrder() {
    return incidentEdgeOrder;
  }

  @Override
  @CanIgnoreReturnValue
  public boolean addNode(N node) {
    checkNotNull(node, "node");

    if (containsNode(node)) {
      return false;
