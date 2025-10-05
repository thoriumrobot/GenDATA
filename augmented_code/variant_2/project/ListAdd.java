/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LTOMLengthOf;

// @skip-test until we bring list support back

    @Positive
public class ListAdd {

    @Positive
  List<Integer> listField;

    @Positive
  void ListAdd(@LTLengthOf("#3") int index, @LTEqLengthOf("#3") int notIndex, List<Integer> list) {
    @Positive
    list.add(index, 4);

    // :: error: (list.access.unsafe.high)
    @Positive
    list.add(notIndex + 1, 4);
    @Positive
  }

    @Positive
  int[] arr = {0};

    @Positive
  void ListAddWrongName(@LTLengthOf("arr") int index, List<Integer> list) {
    // :: error: (list.access.unsafe.high)
    @Positive
    list.add(index, 4);
    @Positive
  }

    @Positive
  void ListAddField() {
    @Positive
    listField.add(listField.size() - 1, 4);
    @Positive
    listField.add(this.listField.size() - 1, 4);
    @Positive
    this.listField.add(listField.size() - 1, 4);
    @Positive
    this.listField.add(this.listField.size() - 1, 4);

    // :: error: (list.access.unsafe.high)
    @Positive
    listField.add(listField.size(), 4);
    // :: error: (list.access.unsafe.high)
    @Positive
    listField.add(this.listField.size(), 4);
    // :: error: (list.access.unsafe.high)
    @Positive
    this.listField.add(listField.size(), 4);
    // :: error: (list.access.unsafe.high)
    @Positive
    this.listField.add(this.listField.size(), 4);
    @Positive
  }

    @Positive
  void ListAddFieldUserAnnotation(@IndexFor("listField") int i) {
    @Positive
    listField.add(i, 4);
    @Positive
    this.listField.add(i, 4);

    // :: error: (list.access.unsafe.high)
    @Positive
    listField.add(i + 4, 4);
    // :: error: (list.access.unsafe.high)
    @Positive
    this.listField.add(i + 4, 4);
    @Positive
  }

    @Positive
  void ListAddUserAnnotation(@IndexFor("#2") int i, List<Integer> list) {
    @Positive
    list.add(i, 4);

    // :: error: (list.access.unsafe.high)
    @Positive
    list.add(i + 4, 4);
    @Positive
  }

    @Positive
  void ListAddUpdateValue(List<Integer> list) {
    @Positive
    @LTEqLengthOf("list") int i = list.size();
    @Positive
    @LTLengthOf("list") int r = list.size() - 1;
    @Positive
    list.add(0);
    @Positive
    @LTLengthOf("list") int k = i;
    @Positive
    @LTOMLengthOf("list") int p = r;
    @Positive
  }

    @Positive
  void ListAddTwo(@LTEqLengthOf({"#2", "#3"}) int i, List<Integer> list, List<Integer> list2) {
    @Positive
    @LTEqLengthOf({"list", "list2"}) int j = i;
    @Positive
    list.add(0);
    // :: error: (list.access.unsafe.high)
    @Positive
    list.get(i);
    // :: error: (list.access.unsafe.high)
    @Positive
    list2.get(i);
    @Positive
  }
    @Positive
}
