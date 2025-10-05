// Source-based slice around line 32
// Method: com.google.common.graph.StandardMutableGraph.backingValueGraph

 * graphs. Instances of this class should be constructed with {@link GraphBuilder}.
 *
 * <p>Time complexities for mutation methods are all O(1) except for {@code removeNode(N node)},
 * which is in O(d_node) where d_node is the degree of {@code node}.
 *
 * @author James Sexton
 * @param <N> Node parameter type
 */
final class StandardMutableGraph<N> extends ForwardingGraph<N> implements MutableGraph<N> {
  private final MutableValueGraph<N, Presence> backingValueGraph;

  /** Constructs a {@link MutableGraph} with the properties specified in {@code builder}. */
  StandardMutableGraph(AbstractGraphBuilder<? super N> builder) {
    this.backingValueGraph = new StandardMutableValueGraph<>(builder);
  }

  @Override
  BaseGraph<N> delegate() {
    return backingValueGraph;
  }
