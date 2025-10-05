// Source-based slice around line 47
// Method: com.google.common.graph.StandardMutableValueGraph.incidentEdgeOrder

 * @author James Sexton
 * @author Joshua O'Madadhain
 * @author Omar Darwish
 * @param <N> Node parameter type
 * @param <V> Value parameter type
 */
final class StandardMutableValueGraph<N, V> extends StandardValueGraph<N, V>
    implements MutableValueGraph<N, V> {

  private final ElementOrder<N> incidentEdgeOrder;

  /** Constructs a mutable graph with the properties specified in {@code builder}. */
  StandardMutableValueGraph(AbstractGraphBuilder<? super N> builder) {
    super(builder);
    incidentEdgeOrder = builder.incidentEdgeOrder.cast();
  }

  @Override
  public ElementOrder<N> incidentEdgeOrder() {
    return incidentEdgeOrder;
