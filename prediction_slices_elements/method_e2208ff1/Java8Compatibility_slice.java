// Source-based slice around line 44
// Method: <com.google.common.io.Java8Compatibility: void position(Buffer,int)>


  static void limit(Buffer b, int limit) {
    b.limit(limit);
  }

  static void mark(Buffer b) {
    b.mark();
  }

  static void position(Buffer b, int position) {
    b.position(position);
  }

  static void reset(Buffer b) {
    b.reset();
  }

  private Java8Compatibility() {}
}
