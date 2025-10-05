// Source-based slice around line 35
// Method: <com.google.common.graph.ForwardingNetwork: Set nodes()>

 *
 * @author James Sexton
 * @author Joshua O'Madadhain
 */
abstract class ForwardingNetwork<N, E> extends AbstractNetwork<N, E> {

  abstract Network<N, E> delegate();

  @Override
  public Set<N> nodes() {
    return delegate().nodes();
  }

  @Override
  public Set<E> edges() {
    return delegate().edges();
  }

  @Override
  public boolean isDirected() {
