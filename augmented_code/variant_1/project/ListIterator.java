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

    @Positive
public class ListIterator {

    @Positive
  List<Integer> listField;

    @Positive
  void ListIterator(
    @Positive
      @LTLengthOf("#3") int index, @LTEqLengthOf("#3") int notIndex, List<Integer> list) {
    @Positive
    list.listIterator(index);

    // :: error: (list.access.unsafe.high)
    @Positive
    list.listIterator(notIndex);
    @Positive
  }

    @Positive
  int[] arr = {0};

    @Positive
  void ListIteratorWrongName(@LTLengthOf("arr") int index, List<Integer> list) {
    // :: error: (list.access.unsafe.high)
    @Positive
    list.listIterator(index);
    @Positive
  }

    @Positive
  void ListIteratorField() {
    @Positive
    listField.listIterator(listField.size() - 1);
    @Positive
    listField.listIterator(this.listField.size() - 1);
    @Positive
    this.listField.listIterator(listField.size() - 1);
    @Positive
    this.listField.listIterator(this.listField.size() - 1);

    // :: error: (list.access.unsafe.high)
    @Positive
    listField.listIterator(listField.size());
    // :: error: (list.access.unsafe.high)
    @Positive
    listField.listIterator(this.listField.size());
    // :: error: (list.access.unsafe.high)
    @Positive
    this.listField.listIterator(listField.size());
    // :: error: (list.access.unsafe.high)
    @Positive
    this.listField.listIterator(this.listField.size());
    @Positive
  }

    @Positive
  void ListIteratorFieldUserAnnotation(@IndexFor("listField") int i) {
    @Positive
    listField.listIterator(i);
    @Positive
    this.listField.listIterator(i);

    // :: error: (list.access.unsafe.high)
    @Positive
    listField.listIterator(1 + i);
    // :: error: (list.access.unsafe.high)
    @Positive
    this.listField.listIterator(1 + i);
    @Positive
  }

    @Positive
  void ListIteratorUserAnnotation(@IndexFor("#2") int i, List<Integer> list) {
    @Positive
    list.listIterator(i);

    // :: error: (list.access.unsafe.high)
    @Positive
    list.listIterator(1 + i);
    @Positive
  }
    @Positive
}
