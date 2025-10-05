// Source-based slice around line 38
// Method: com.google.common.collect.Cut.endpoint

 * "numbers") into two sections; this can be done below a certain value, above a certain value,
 * below all values or above all values. With this object defined in this way, an interval can
 * always be represented by a pair of {@code Cut} instances.
 *
 * @author Kevin Bourrillion
 */
@SuppressWarnings("rawtypes") // https://github.com/google/guava/issues/989
@GwtCompatible
abstract class Cut<C extends Comparable> implements Comparable<Cut<C>>, Serializable {
  final C endpoint;

  Cut(C endpoint) {
    this.endpoint = endpoint;
  }

  abstract boolean isLessThan(C value);

  abstract BoundType typeAsLowerBound();

  abstract BoundType typeAsUpperBound();
