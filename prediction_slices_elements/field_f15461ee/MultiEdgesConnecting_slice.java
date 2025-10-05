// Source-based slice around line 41
// Method: com.google.common.graph.MultiEdgesConnecting.targetNode

 * <p>The {@link #outEdgeToNode} map allows this class to work on networks with parallel edges. See
 * {@link EdgesConnecting} for a class that is more efficient but forbids parallel edges.
 *
 * @author James Sexton
 * @param <E> Edge parameter type
 */
abstract class MultiEdgesConnecting<E> extends AbstractSet<E> {

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
