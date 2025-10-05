// Source-based slice around line 30
// Method: <com.google.common.cache.LongAddable: long sum()>

 *
 * @author Louis Wasserman
 */
@GwtCompatible
interface LongAddable {
  void increment();

  void add(long x);

  long sum();
}
