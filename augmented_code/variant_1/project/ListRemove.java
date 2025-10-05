/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;

// @skip-test until we bring list support back
// @skip-test can't handle until TreeUtils.getMethod has a way to precisely handle method
// overloading

    @Positive
public class ListRemove {

    @Positive
  List<Integer> listField;

    @Positive
  void ListRemove(
    @Positive
      @LTLengthOf("#3") int index, @LTEqLengthOf("#3") int notIndex, List<Integer> list) {
    @Positive
    list.remove(index);

    // :: error: (list.access.unsafe.high)
    @Positive
    list.remove(notIndex);
    @Positive
  }

    @Positive
  void ListRemoveWrongName(@LTLengthOf("arr") int index, List<Integer> list) {
    // :: error: (list.access.unsafe.high)
    @Positive
    list.remove(index);
    @Positive
  }

    @Positive
  void ListRemoveField() {
    @Positive
    listField.remove(listField.size() - 1);
    @Positive
    listField.remove(this.listField.size() - 1);
    @Positive
    this.listField.remove(listField.size() - 1);
    @Positive
    this.listField.remove(this.listField.size() - 1);

    // :: error: (list.access.unsafe.high)
    @Positive
    listField.remove(listField.size());
    // :: error: (list.access.unsafe.high)
    @Positive
    listField.remove(this.listField.size());
    // :: error: (list.access.unsafe.high)
    @Positive
    this.listField.remove(listField.size());
    // :: error: (list.access.unsafe.high)
    @Positive
    this.listField.remove(this.listField.size());
    @Positive
  }

    @Positive
  void ListRemoveFieldUserAnnotation(@IndexFor("listField") int i) {
    @Positive
    listField.remove(i);
    @Positive
    this.listField.remove(i);

    // :: error: (list.access.unsafe.high)
    @Positive
    listField.remove(i + 1);
    // :: error: (list.access.unsafe.high)
    @Positive
    this.listField.remove(i + 1);
    @Positive
  }

    @Positive
  void ListRemoveUserAnnotation(@IndexFor("list") int i, List<Integer> list) {
    @Positive
    list.remove(i);

    // :: error: (list.access.unsafe.high)
    @Positive
    list.remove(i + 1);
    // :: error: (list.access.unsafe.high)
    @Positive
    list.remove(i);
    @Positive
  }

    @Positive
  void FailRemove(List<Integer> list) {
    @Positive
    @LTLengthOf("list") int i = list.size() - 1;
    @Positive
    try {
    @Positive
      list.remove(1);
    @Positive
    } catch (Exception e) {
    @Positive
    }

    @Positive
    @LTLengthOf("list") int m = i;
    @Positive
  }

    @Positive
  void RemoveUpdate(List<Integer> list) {
    @Positive
    int m = list.size() - 1;
    @Positive
    list.get(m);
    @Positive
    list.remove(m);
    // :: error: (list.access.unsafe.high)
    @Positive
    list.get(m);
    @Positive
  }
    @Positive
}
