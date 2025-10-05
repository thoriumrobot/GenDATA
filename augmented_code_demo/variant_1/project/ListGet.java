    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;

// @skip-test until we bring list support back

    @Positive
public class ListGet {

    @Positive
  List<Integer> listField;
    @Positive
  int[] arr = {0};

    @Positive
  void ListGet(@LTLengthOf("#3") int index, @LTEqLengthOf("#3") int notIndex, List<Integer> list) {
    @Positive
    list.get(index);

    // :: error: (list.access.unsafe.high)
    @Positive
    list.get(notIndex);
    @Positive
  }

    @Positive
  void ListGetWrongName(@LTLengthOf("arr") int index, List<Integer> list) {
    // :: error: (list.access.unsafe.high)
    @Positive
    list.get(index);
    @Positive
  }

    @Positive
  void ListGetField() {
    @Positive
    listField.get(listField.size() - 1);
    @Positive
    listField.get(this.listField.size() - 1);
    @Positive
    this.listField.get(listField.size() - 1);
    @Positive
    this.listField.get(this.listField.size() - 1);

    // :: error: (list.access.unsafe.high)
    @Positive
    listField.get(listField.size());
    // :: error: (list.access.unsafe.high)
    @Positive
    listField.get(this.listField.size());
    // :: error: (list.access.unsafe.high)
    @Positive
    this.listField.get(listField.size());
    // :: error: (list.access.unsafe.high)
    @Positive
    this.listField.get(this.listField.size());
    @Positive
  }

    @Positive
  void ListGetFieldUserAnnotation(@IndexFor("listField") int i) {
    @Positive
    listField.get(i);
    @Positive
    this.listField.get(i);

    // :: error: (list.access.unsafe.high)
    @Positive
    listField.get(i + 1);
    // :: error: (list.access.unsafe.high)
    @Positive
    this.listField.get(i + 1);
    @Positive
  }

    @Positive
  void ListGetUserAnnotation(@IndexFor("#2") int i, List<Integer> list) {
    @Positive
    list.get(i);

    // :: error: (list.access.unsafe.high)
    @Positive
    list.get(i + 1);
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
