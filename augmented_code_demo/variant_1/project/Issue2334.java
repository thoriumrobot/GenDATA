// Test case for issue #2334: http://tinyurl.com/cfissue/2334

    @Positive
import org.checkerframework.checker.index.qual.NonNegative;

    @Positive
public class Issue2334 {

    @Positive
  void hasSideEffect() {}

    @Positive
  String stringField;

    @Positive
  void m1(String stringFormal) {
    @Positive
    if (stringFormal.indexOf('d') != -1) {
    @Positive
      hasSideEffect();
    @Positive
      @NonNegative int i = stringFormal.indexOf('d');
    @Positive
    }
    @Positive
  }

    @Positive
  void m2() {
    @Positive
    if (stringField.indexOf('d') != -1) {
    @Positive
      hasSideEffect();
      // :: error: (assignment)
    @Positive
      @NonNegative int i = stringField.indexOf('d');
    @Positive
    }
    @Positive
  }

    @Positive
  void m3(String stringFormal) {
    @Positive
    if (stringFormal.indexOf('d') != -1) {
    @Positive
      System.out.println("hey");
    @Positive
      @NonNegative int i = stringFormal.indexOf('d');
    @Positive
    }
    @Positive
  }

    @Positive
  void m4() {
    @Positive
    if (stringField.indexOf('d') != -1) {
    @Positive
      System.out.println("hey");
      // :: error: (assignment)
    @Positive
      @NonNegative int i = stringField.indexOf('d');
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
