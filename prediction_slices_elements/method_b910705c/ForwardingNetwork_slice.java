// Source-based slice around line 32
// Method: <com.google.common.graph.ForwardingNetwork: Network delegate()>

/**
 * A class to allow {@link Network} implementations to be backed by a provided delegate. This is not
 * currently planned to be released as a general-purpose forwarding class.
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
