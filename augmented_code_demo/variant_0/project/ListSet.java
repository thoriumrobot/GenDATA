    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;

// @skip-test until we bring list support back

    @Positive
public class ListSet {

    @Positive
  List<Integer> listField;

    @Positive
  void ListSet(@LTLengthOf("#3") int index, @LTEqLengthOf("#3") int notIndex, List<Integer> list) {
    @Positive
    list.set(index, 4);

    // :: error: (list.access.unsafe.high)
    @Positive
    list.set(notIndex, 4);
    @Positive
  }

    @Positive
  int[] arr = {0};

    @Positive
  void ListSetWrongName(@LTLengthOf("arr") int index, List<Integer> list) {
    // :: error: (list.access.unsafe.high)
    @Positive
    list.set(index, 4);
    @Positive
  }

    @Positive
  void ListSetField() {
    @Positive
    listField.set(listField.size() - 1, 4);
    @Positive
    listField.set(this.listField.size() - 1, 4);
    @Positive
    this.listField.set(listField.size() - 1, 4);
    @Positive
    this.listField.set(this.listField.size() - 1, 4);

    // :: error: (list.access.unsafe.high)
    @Positive
    listField.set(listField.size(), 4);
    // :: error: (list.access.unsafe.high)
    @Positive
    listField.set(this.listField.size(), 4);
    // :: error: (list.access.unsafe.high)
    @Positive
    this.listField.set(listField.size(), 4);
    // :: error: (list.access.unsafe.high)
    @Positive
    this.listField.set(this.listField.size(), 4);
    @Positive
  }

    @Positive
  void ListSetFieldUserAnnotation(@IndexFor("listField") int i) {
    @Positive
    listField.set(i, 4);
    @Positive
    this.listField.set(i, 4);

    // :: error: (list.access.unsafe.high)
    @Positive
    listField.set(i + 1, 4);
    // :: error: (list.access.unsafe.high)
    @Positive
    this.listField.set(i + 1, 4);
    @Positive
  }

    @Positive
  void ListSetUserAnnotation(@IndexFor("#2") int i, List<Integer> list) {
    @Positive
    list.set(i, 4);

    // :: error: (list.access.unsafe.high)
    @Positive
    list.set(i + 1, 4);
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
