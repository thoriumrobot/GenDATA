// Source-based slice around line 26
// Method: <com.google.common.cache.LongAddable: void increment()>

import com.google.common.annotations.GwtCompatible;

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
