// Source-based slice around line 36
// Method: <com.google.common.io.Java8Compatibility: void limit(Buffer,int)>

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

  static void mark(Buffer b) {
    b.mark();
  }

  static void position(Buffer b, int position) {
    b.position(position);
  }
