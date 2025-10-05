// Source-based slice around line 28
// Method: <com.google.common.cache.LongAddable: void add(long)>

/**
 * Abstract interface for objects that can concurrently add longs.
 *
 * @author Louis Wasserman
 */
@GwtCompatible
interface LongAddable {
  void increment();

  void add(long x);

  long sum();
}
