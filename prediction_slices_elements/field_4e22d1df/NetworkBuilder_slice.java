// Source-based slice around line 77
// Method: com.google.common.graph.NetworkBuilder.allowsParallelEdges

 *     unless it is constrained by using a method like {@link #nodeOrder}, or the builder is
 *     constructed based on an existing {@code Network} using {@link #from(Network)}.
 * @param <E> The most general edge type this builder will support. This is normally {@code Object}
 *     unless it is constrained by using a method like {@link #edgeOrder}, or the builder is
 *     constructed based on an existing {@code Network} using {@link #from(Network)}.
 * @since 20.0
 */
@Beta
public final class NetworkBuilder<N, E> extends AbstractGraphBuilder<N> {
  boolean allowsParallelEdges = false;
  ElementOrder<? super E> edgeOrder = ElementOrder.insertion();
  Optional<Integer> expectedEdgeCount = Optional.absent();

  /** Creates a new instance with the specified edge directionality. */
  private NetworkBuilder(boolean directed) {
    super(directed);
  }

  /** Returns a {@link NetworkBuilder} for building directed networks. */
  public static NetworkBuilder<Object, Object> directed() {
