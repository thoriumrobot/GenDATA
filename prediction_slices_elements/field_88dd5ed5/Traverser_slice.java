// Source-based slice around line 68
// Method: com.google.common.graph.Traverser.successorFunction

 * @author Jens Nyman
 * @param <N> Node parameter type
 * @since 23.1
 */
@Beta
@DoNotMock(
    "Call forGraph or forTree, passing a lambda or a Graph with the desired edges (built with"
        + " GraphBuilder)")
public abstract class Traverser<N> {
  private final SuccessorsFunction<N> successorFunction;

  private Traverser(SuccessorsFunction<N> successorFunction) {
    this.successorFunction = checkNotNull(successorFunction);
  }

  /**
   * Creates a new traverser for the given general {@code graph}.
   *
   * <p>Traversers created using this method are guaranteed to visit each node reachable from the
   * start node(s) at most once.
