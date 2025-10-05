// Source-based slice around line 32
// Method: <com.google.common.base.Java8Compatibility: void flip(Buffer)>

 * https://github.com/google/guava/issues/3990
 */
@J2ktIncompatible
@GwtIncompatible
final class Java8Compatibility {
  static void clear(Buffer b) {
    b.clear();
  }

  static void flip(Buffer b) {
    b.flip();
  }

  static void limit(Buffer b, int limit) {
    b.limit(limit);
  }

  static void position(Buffer b, int position) {
    b.position(position);
  }
